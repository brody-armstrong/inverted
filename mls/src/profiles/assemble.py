t=open('src/profiles/template.html').read()
t=t.replace('__DATA__',open('out/data.json').read()).replace('__PATHWAY__',open('out/pathway.json').read()).replace('__TEAMS__',open('out/teams.json').read())
open('out/index.html','w').write(t)
