"""Whole-program register use for gen.py --abi-trust: what each lifted function reads of the registers it is entered
with, what its callers read of the registers it returns with, and what it always writes.

gen.py --abi-trust keeps every general register of a function in a C local and syncs a local with the Cpu only where
something on the other side reads it. That needs to know the other side:

  IN[f]    registers f may read before writing them, on some path, itself or through the functions it calls or jumps
           to. A caller stores only those (of the ones it changed) before calling f.
  OUT[f]   registers some caller of f may read after f returns, before writing them. f stores only those (of the ones
           it changed, and the callee-saved ones) before it returns, and a caller loads again after the call only
           what its own code reads.
  MUST[f]  registers f writes on every path to a return, NORET[f] when it has none: a register a caller reads after
           the call need not be current before it when the callee always writes it.

They are the fixed point of a liveness over every function at once (solve()): backward inside a function, IN flowing
from callee to caller and OUT from caller to callee. gen.py runs each function's liveness again with the final
tables (live()) for the registers live after each of its calls.

Between two functions joined by a direct call or branch nothing assumes AAPCS64: a read is a read. A push of a
callee-saved register keeps it current, a register read after a call is returned and loaded again whichever it is,
and a function that takes a value in X9-X17 gets it. Reads of known code are the `real` ones below.

The target of an indirect call or jump (blr, br, a call through rt_dispatch: also every host function) is unknown,
so its reads are `assumed`: X0-X8, X19-X30 and SP (IND_USE). So are the reads behind an unresolved computed branch
of an all-label function, which may go to any of its instructions. An assumed read keeps a register current back to the
function's entry and from there in its callers, so arguments passed down through any number of functions arrive.
It does not reach back across an earlier call for X0-X17: an unknown callee is not taken to read what another callee
left in a register AAPCS64 lets that callee change (X0 and X1 are returned by every function anyway). The same holds
for a patch helper, which reads the registers its C source names (helper_regs()), and for X9-X17 of an indirect
target: the function making the call stores every register it changed, so what it wrote itself is current, but an
older value in X9-X17 is not kept for the target. report() counts what the lifted code does against these rules.

Every function may be the target of an indirect call, so every function returns X0 and X1 (the runtime reads them
after rt_call too) and what the code after any indirect call reads itself before its next call (out_ind: nothing in
Astral Chain 1.0.1). A register that only a later callee reads, or a caller further up, an unknown target is not taken
to return: such reads exist only behind calls that do not return (an error handler called through a pointer, with
another block or the next function after it) and in callees that read an argument register they were not passed. A callee preserves X19-X29 and SP
(recomp/regs.py; not with gen.py --no-callee-saved-trust, where kept is empty).

Registers are bit masks: bit n for Xn, bit 31 for SP. X18 never is a local and is left out.
"""
import json
import re
from array import array
from pathlib import Path

SP = 31
X18 = 1 << 18
ALL = 0xffffffff & ~X18
TEMPS = sum(1 << r for r in range(9, 18))
IND_USE = ALL & ~TEMPS
OUT_BASE = 0b11
SVC_USE = 0xff

# Events of an instruction (gen.py Emitter.function): what it does besides its own reads and writes.
CALL, TAIL, RET, TRAP, SVC, HELPER = range(6)
PLT = 'plt'   # a TAIL through an import stub, resolved to its function in another module (gen.py import_stubs())


def mask(regs):
    """The bits of the registers `regs`; a register named twice counts once (a patch helper's source names a register
    at each use: a sum gave c->x[0] read and written as x1)."""
    m = 0
    for r in regs:
        m |= 1 << r
    return m


def regs_of(m):
    return {r for r in range(32) if m >> r & 1}


def names(m):
    return ' '.join('sp' if r == SP else f'x{r}' for r in sorted(regs_of(m))) or 'none'


class Summary:
    """One function for the analysis. use / dfn: per instruction, the registers its C reads, and the ones it always
    writes; succ: successors inside the function (gen.py flow(); the index after the last instruction stands for
    falling into the next function), kept as {instruction: successors} for the instructions that do not just go on
    to the next one; anywhere: the unresolved computed branches of an all-label function, which may go to any of its
    instructions; ev: {instruction: [(event, target address or None, helper name or PLT or None)]}; patched: instructions
    that are a patched site (alternatives: nothing is always written, every event may or may not happen). plain: an
    all-label function, whose registers all stay in the Cpu."""
    __slots__ = ('entry', 'n', 'use', 'dfn', 'succ', 'anywhere', 'ev', 'patched', 'plain', '_nodes')

    def __init__(self, entry, use, dfn, succ, anywhere, ev, patched, plain):
        self.entry, self.n, self.plain = entry, len(use), plain
        self.use, self.dfn = array('I', use), array('I', dfn)
        self.succ = {i: tuple(x) for i, x in enumerate(succ) if x != [i + 1]}
        self.anywhere, self.ev, self.patched = frozenset(anywhere), ev, patched
        self._nodes = None

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__ if k != '_nodes'}

    def __setstate__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self._nodes = None

    def nodes(self):
        """The function as the liveness walks it: (first instruction of each node, use, dfn, successor nodes, node of
        each event instruction). A run of instructions with no event, no branch and no branch target is one node,
        with the reads and writes of the run; an event instruction is a node of its own. A function with a computed
        branch that may go to any instruction keeps one node per instruction."""
        if self._nodes is None:
            n, succ, ev = self.n, self.succ, self.ev
            if self.anywhere:
                lead = set(range(n))
            else:
                lead = {0}
                for i, ss in succ.items():
                    lead.add(i + 1)
                    lead.update(ss)
                for i in ev:
                    lead.add(i)
                    lead.add(i + 1)
            first = sorted(x for x in lead if x < n)
            node_of = {a: k for k, a in enumerate(first)}
            use, dfn, nsucc = [], [], []
            for k, a in enumerate(first):
                b = first[k + 1] if k + 1 < len(first) else n
                u = d = 0
                for j in range(a, b):
                    u |= self.use[j] & ~d
                    d |= self.dfn[j]
                use.append(u)
                dfn.append(d)
                nsucc.append(tuple(node_of[x] for x in succ.get(b - 1, (b,)) if x < n))
            self._nodes = first, use, dfn, nsucc, {i: node_of[i] for i in ev}
        return self._nodes

    def targets(self):
        """The direct call and jump targets."""
        return {t for evs in self.ev.values() for k, t, _ in evs if k in (CALL, TAIL) and t is not None}


class Tables:
    def __init__(self, kept, helpers=None):
        self.kept = kept          # registers a call leaves as they were (mask)
        self.INR, self.INH = {}, {}   # IN: the real and the assumed reads
        self.OUT, self.MUST, self.NORET = {}, {}, {}
        self.out_ind = 0          # really read after some indirect call: every function returns them
        self.helpers = helpers or {}   # {patch helper name: registers it reads}

    @classmethod
    def load(cls, path):
        """The tables report() wrote."""
        d = json.loads(Path(path).read_text())
        T = cls(d['kept'], d['helpers'])
        T.out_ind = d['out_ind']
        for f, (inr, inh, out, must_, noret) in d['functions'].items():
            f = int(f, 16)
            T.INR[f], T.INH[f], T.OUT[f], T.MUST[f], T.NORET[f] = inr, inh, out, must_, bool(noret)
        return T

    def out(self, f):
        return self.OUT.get(f, 0) | self.out_ind | OUT_BASE

    def inn(self, f):
        return self.INR[f] | self.INH[f]

    def helper(self, name):
        return self.helpers.get(name, ALL)


def before(S, i, u, d, lo, ro, po, ho, T, out_f):
    """(local, real, returned, assumed) liveness before the node of S that starts at instruction i, reads u and
    always writes d, from the liveness after it.
    local: registers the function's own code reads, in its locals;
    real: registers known code reads, the function's or another's: at the entry, the function's IN;
    returned: registers a caller reads after the function returns (its OUT), from the return back to where the
        function or a callee last writes them. Whoever writes one stores it, and the caller that reads it made it
        current before its call: it asks nothing of the function's other callers, so it is no part of IN;
    assumed: registers an unknown target or a patch helper is taken to read."""
    evs = S.ev.get(i)
    if not evs:
        return u | (lo & ~d), u | (ro & ~d), po & ~d, ho & ~d
    kept = T.kept
    if i in S.patched:   # any alternative may run: no kill, every event's reads
        r, p, h = u | ro, po, ho
        for k, t, how in evs:
            if k == CALL:
                if t is None:
                    h |= IND_USE
                else:
                    r |= T.INR[t]
                    h |= T.INH[t]
            elif k == TAIL:
                p |= out_f
                if t is None or how == PLT:
                    h |= IND_USE
                if t is not None:
                    r |= T.INR[t]
                    h |= T.INH[t]
            elif k == RET:
                p |= out_f
            elif k == SVC:
                r |= SVC_USE
            elif k == HELPER:
                h |= T.helper(how)
        return u | lo, r, p, h
    k, t, how = evs[0]
    if k == CALL:
        # The call's own writes (X30) come before the callee's reads. A local the callee may change is loaded again
        # after the call if the function reads it, so it is not live before; one the callee keeps passes through.
        if t is None:
            return u | (lo & kept & ~d), u | (ro & ~d), po & ~d, (IND_USE | (ho & kept)) & ~d
        if T.NORET[t]:
            return u, u | (T.INR[t] & ~d), 0, T.INH[t] & ~d
        through = kept | ~T.MUST[t]
        return (u | (lo & kept & ~d), u | ((T.INR[t] | (ro & through)) & ~d), po & through & ~d,
                (T.INH[t] | (ho & kept)) & ~d)
    if k == TAIL:   # lo, ro, po, ho: the fall-through of a conditional branch
        if t is None:
            return u | lo, u | ro, po | out_f, ho | IND_USE
        if how == PLT:   # an import stub: the function of the other module, or a host function in its place
            return u | lo, u | ro | T.INR[t], po | (0 if T.NORET[t] else out_f), ho | T.INH[t] | IND_USE
        return u | lo, u | ro | T.INR[t], po | (0 if T.NORET[t] else out_f & ~T.MUST[t]), ho | T.INH[t]
    if k == RET:
        return u | lo, u | ro, po | out_f, ho
    if k == SVC:   # every local is loaded again afterwards: what the code reads then must be current in the Cpu
        return u, u | ro | SVC_USE, po, ho
    return u | lo, u | ro, po, ho   # TRAP


def live(S, T):
    """({event instruction: local liveness after it}, the same for the real and for the returned liveness), and the
    (real, assumed) liveness at the entry of S."""
    first, use, dfn, nsucc, ev_node = S.nodes()
    n, out_f = len(first), T.out(S.entry)
    li, ri, pi, hi = [0] * n, [0] * n, [0] * n, [0] * n
    lo, ro, po = [0] * n, [0] * n, [0] * n
    anywhere = S.anywhere
    # An unresolved computed branch may go to any instruction of its function: it is taken to read what any of them
    # reads (an assumed read: the code it reaches reads the Cpu directly, so what the function wrote itself is there).
    every = 0
    changed = True
    while changed:
        changed = False
        for k in range(n - 1, -1, -1):
            a = b = p = c = 0
            for t in nsucc[k]:
                a |= li[t]
                b |= ri[t]
                p |= pi[t]
                c |= hi[t]
            if first[k] in anywhere:
                c |= every
            lo[k], ro[k], po[k] = a, b, p
            x = before(S, first[k], use[k], dfn[k], a, b, p, c, T, out_f)
            if x != (li[k], ri[k], pi[k], hi[k]):
                li[k], ri[k], pi[k], hi[k] = x
                changed = True
        if anywhere:
            x = 0
            for k in range(n):
                x |= li[k] | ri[k] | pi[k] | hi[k]
            if x != every:
                every = x
                changed = True
    return ({i: lo[k] for i, k in ev_node.items()}, {i: ro[k] for i, k in ev_node.items()},
            {i: po[k] for i, k in ev_node.items()}, (ri[0], hi[0]))


def must(S, T):
    """(registers S writes on every path to a return, whether it has no path to one)"""
    if S.plain:
        return 0, False
    first, use, dfn, nsucc, ev_node = S.nodes()
    state = [None] * len(first)   # None: not reached
    state[0] = 0
    work = [0]
    while work:
        k = work.pop()
        cur, i = state[k], first[k]
        evs = S.ev.get(i)
        nxt = cur | dfn[k]
        if evs:
            if i in S.patched:
                nxt = cur
            else:
                e, t, _ = evs[0]
                if e == CALL:
                    if t is not None:
                        if T.NORET[t]:
                            continue
                        nxt |= T.MUST[t]
                elif e == TAIL:
                    nxt = cur
                elif e in (RET, TRAP):
                    continue
        for t in nsucc[k]:
            old = state[t]
            new = nxt if old is None else old & nxt
            if new != old:
                state[t] = new
                work.append(t)
    m, any_exit = ALL, False
    for i, evs in S.ev.items():
        cur = state[ev_node[i]]
        if cur is None:
            continue
        for e, t, how in evs:
            if e == RET or (e == TAIL and (t is None or i in S.patched)):
                m &= cur
                any_exit = True
            elif e == TAIL and not T.NORET[t]:
                m &= cur if how == PLT else cur | T.MUST[t]
                any_exit = True
    return (m, False) if any_exit else (ALL, True)


def solve(summaries, kept, helpers=None, log=None):
    """Tables for the functions of summaries ({entry: Summary}, every lifted module together)."""
    T = Tables(kept, helpers)
    callers = {f: set() for f in summaries}   # who must be looked at again when f's tables change
    for f, S in summaries.items():
        T.MUST[f], T.NORET[f] = ALL, True   # the greatest fixed point: start from "never returns"
        T.INR[f] = T.INH[f] = T.OUT[f] = 0
        for t in S.targets():
            if t in summaries:
                callers[t].add(f)
            else:   # a target that was not summarised (gen.py --limit): anything
                T.MUST[t], T.NORET[t], T.INR[t], T.INH[t], T.OUT[t] = 0, False, ALL, ALL, 0

    def run(step):
        work, queued, steps = list(summaries), set(summaries), 0
        while work:
            f = work.pop()
            queued.discard(f)
            steps += 1
            for g in step(f):
                if g in summaries and g not in queued:
                    queued.add(g)
                    work.append(g)
        return steps

    def step_must(f):
        new = must(summaries[f], T)
        if new == (T.MUST[f], T.NORET[f]):
            return ()
        T.MUST[f], T.NORET[f] = new
        return callers[f]

    steps = run(step_must)
    if log:
        log(f'ipa: must-write sets in {steps} function passes, {sum(T.NORET[f] for f in summaries)} functions never '
            f'return')
    grew = []
    T.why = why = {}   # {(function or None for every function, register): (function, instruction, how)}: the first
                       # reason the function returns the register (report())

    def note(t, m, f, i, how):
        for r in regs_of(m):
            why.setdefault((t, r), (f, i, how))

    def step_live(f):
        S = summaries[f]
        lo, rr, rp, at_entry = live(S, T)
        ro = {i: rr[i] | rp[i] for i in rr}
        again = set()
        if at_entry != (T.INR[f], T.INH[f]):
            T.INR[f], T.INH[f] = at_entry
            again |= callers[f]
        out_f = T.out(f)
        for i, evs in S.ev.items():
            for k, t, _ in evs:
                if k == CALL:
                    o = ro[i] & ~kept
                    if t is None:
                        # An unknown callee returns X0 and X1, and what the code after the call reads itself. What
                        # only a later callee reads, or a caller of this function, it is not taken to return.
                        o = lo[i] & ~kept & ~OUT_BASE
                        if o & ~T.out_ind:
                            grew.append((o & ~T.out_ind, f'read after the indirect call at {f:#x}+{4 * i:#x}'))
                            note(None, o & ~T.out_ind, f, i, 'read')
                            T.out_ind |= o
                    elif o & ~T.OUT[t]:
                        note(t, o & ~T.OUT[t] & rr[i], f, i, 'read')
                        note(t, o & ~T.OUT[t], f, i, 'returned')
                        T.OUT[t] |= o
                        again.add(t)
                elif k == TAIL:
                    if t is None:
                        pass   # an unknown target: see the indirect call above
                    elif out_f & ~T.OUT[t]:
                        note(t, out_f & ~T.OUT[t], f, i, 'jump')
                        T.OUT[t] |= out_f
                        again.add(t)
        return again

    steps = rounds = 0
    while True:
        rounds += 1
        n_grew = len(grew)
        steps += run(step_live)
        if len(grew) == n_grew:   # else every function returns more: all of them again
            break
    if log:
        for m, why in grew:
            log(f'ipa: every function returns {names(m)}: {why}')
        log(f'ipa: register use in {steps} function passes, {rounds} rounds')
    return T


HELPER_DEF = r'^[\w ]*\b%s\(Cpu \*c\)\s*\{'


def helper_regs(name, sources):
    """Registers the patch helper `name` reads, from its C source: every c->x[n] and c->sp it names. ALL when it is
    not found, indexes the registers with something else than a number, or passes the Cpu on to another function."""
    for text in sources:
        m = re.search(HELPER_DEF % re.escape(name), text, re.M)
        if not m:
            continue
        # the body ends at its closing brace; braces in comments and in string and character literals ('{') do not count
        d, i, n = 1, m.end(), len(text)
        while d and i < n:
            ch = text[i]
            if ch in '"\'':
                i += 1
                while i < n and text[i] != ch:
                    i += 2 if text[i] == '\\' else 1
            elif text.startswith('//', i):
                i = text.find('\n', i)
                i = n if i < 0 else i
            elif text.startswith('/*', i):
                i = text.find('*/', i + 2)
                i = n if i < 0 else i + 1
            else:
                d += {'{': 1, '}': -1}.get(ch, 0)
            i += 1
        if d:
            return ALL
        body = text[m.end():i - 1]
        if re.search(r'c->x\[(?!\d+\])', body) or re.search(r'(?<!->)\bc\b(?!->)', body):
            return ALL
        return (mask(int(x) for x in re.findall(r'c->x\[(\d+)\]', body)) | (1 << SP if 'c->sp' in body else 0)) & ALL
    return ALL


def load_helpers(names_, runtime_dir):
    sources = [p.read_text(encoding='utf-8', errors='replace') for p in sorted(Path(runtime_dir).glob('*.c'))]
    return {n: helper_regs(n, sources) for n in names_}


def report(summaries, T, path, log):
    """Counts for the log and the tables as JSON (tests/diff/func.py and tools read them)."""
    n = len(summaries)
    temps_in = [f for f in summaries if T.inn(f) & TEMPS]
    more_out = [f for f in summaries if T.OUT[f] & ~OUT_BASE & ~T.out_ind]
    hist = [0] * 33
    for f in summaries:
        hist[bin(T.OUT[f] & ~OUT_BASE & ~T.out_ind & ~T.kept).count('1')] += 1
    log('ipa: functions by number of registers returned besides X0, X1: ' +
        ', '.join(f'{k}: {v}' for k, v in enumerate(hist) if v))
    per = [0] * 32
    for f in summaries:
        for r in regs_of(T.OUT[f] & ~OUT_BASE & ~T.out_ind & ~T.kept):
            per[r] += 1
    log('ipa: functions returning each: ' + ', '.join(f'x{r}: {v}' for r, v in enumerate(per) if v))
    per = [0] * 32
    for f in summaries:
        for r in regs_of(T.inn(f)):
            per[r] += 1
    log('ipa: functions reading each at entry: ' + ', '.join(f'{"sp" if r == 31 else "x" + str(r)}: {v}' for r, v in enumerate(per) if v))
    log(f'ipa: {n} functions ({sum(s.plain for s in summaries.values())} all-label); read at entry: '
        f'{sum(bin(T.inn(f)).count("1") for f in summaries) / n:.1f} registers on average, X9-X17 in '
        f'{len(temps_in)} functions; returned besides X0 and X1: {names(T.out_ind & ~OUT_BASE)} by every function, '
        f'more by {len(more_out)}; patch helpers taken to read every register: '
        f'{sorted(k for k, v in T.helpers.items() if v == ALL) or "none"}')
    path.write_text(json.dumps({
        'kept': T.kept, 'out_ind': T.out_ind, 'helpers': T.helpers,
        'functions': {f'{f:x}': [T.INR[f], T.INH[f], T.out(f), T.MUST[f], int(T.NORET[f])] for f in sorted(summaries)}}))
