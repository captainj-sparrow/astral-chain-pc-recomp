import sys; sys.path.insert(0, 'recomp')
from layout import layout, write, base_of, romfs_size
exefs = '/home/utente/Games/Astral Chain/exefs'
write(layout(exefs, 'v101'), 'generated/v101',
      romfs_size(exefs), base_of('v101'))
print(open('generated/v101/layout.json').read())