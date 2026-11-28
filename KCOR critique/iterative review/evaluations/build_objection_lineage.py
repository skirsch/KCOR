from pathlib import Path
import re,json
base=Path('KCOR critique');ob=base/'objections'
families={
'Reference-frailty estimand, scope and benchmark':['1.1','1.5','1.6','2.1','2.6','2.8','4.8','5.3','6.6','6.7','6.9','7.11','8.6','8.12','8.13'],
'Nonlinear wave correction, entry and alpha':['1.2','2.4','3.2','4.1','4.2','4.3','4.4','4.7','5.1','5.10','6.12','6.13','7.10','8.5'],
'Frailty fitting, information and numerical safeguards':['1.3','2.2','2.3','2.7','3.1','4.5','5.2','5.9','5.11','6.5','6.11','7.1','7.3','7.9','8.4','8.11'],
'Population, risk sets, aggregation and uncertainty delivery':['2.9','4.6','5.4','5.5','5.6','5.7','6.1','6.2','6.14','7.2','7.4','7.5','8.1','8.2','8.3'],
'Validation design, diagnostics, provenance and reproducibility':['1.4','2.5','2.10','3.3','4.9','5.8','5.12','6.3','6.4','6.8','6.10','7.6','7.7','7.8','8.7','8.8','8.9','8.10']
}
# Direct return edges only when documented. Related edges are not claims of recurrence.
returns={'6.1':['5.7'],'6.7':['2.1','1.5'],'6.13':['5.10'],'6.14':['5.6'],'7.5':['6.14'],'7.6':['6.3'],'7.9':['3.2'],'7.10':['6.13'],'7.11':['6.7'],'8.1':['7.2'],'8.9':['7.6'],'8.10':['7.8'],'8.11':['7.9'],'8.12':['7.11']}
families['Population, risk sets, aggregation and uncertainty delivery'] += ['9.1','9.2','9.5','9.6','9.8','9.9','9.16']
families['Frailty fitting, information and numerical safeguards'] += ['9.4','9.11','9.13']
families['Nonlinear wave correction, entry and alpha'] += ['9.7','9.12','9.15']
families['Validation design, diagnostics, provenance and reproducibility'] += ['9.3','9.10','9.14']
returns.update({'9.1':['8.1'],'9.2':['8.3'],'9.3':['8.10'],'9.4':['8.13'],'9.16':['6.14']})
families['Population, risk sets, aggregation and uncertainty delivery'] += ['10.1','10.6','10.8','10.10']
families['Frailty fitting, information and numerical safeguards'] += ['10.2','10.3']
families['Nonlinear wave correction, entry and alpha'] += ['10.4','10.5']
families['Reference-frailty estimand, scope and benchmark'] += ['10.7']
families['Validation design, diagnostics, provenance and reproducibility'] += ['10.9']
returns.update({'10.1':['9.1'],'10.2':['9.4'],'10.10':['9.16']})
families['Population, risk sets, aggregation and uncertainty delivery'] += ['11.1', '11.2', '11.3', '11.6', '11.7', '11.8', '11.13', '11.16']
families['Validation design, diagnostics, provenance and reproducibility'] += ['11.4', '11.9', '11.12', '11.17']
families['Nonlinear wave correction, entry and alpha'] += ['11.5', '11.14']
families['Reference-frailty estimand, scope and benchmark'] += ['11.10', '11.11', '11.15']
families['Frailty fitting, information and numerical safeguards'] += ['11.18']
families['Population, risk sets, aggregation and uncertainty delivery'] += ['12.1', '12.7', '12.8', '12.9']
families['Validation design, diagnostics, provenance and reproducibility'] += ['12.5', '12.11']
families['Nonlinear wave correction, entry and alpha'] += ['12.2']
families['Reference-frailty estimand, scope and benchmark'] += ['12.3', '12.6', '12.10']
families['Frailty fitting, information and numerical safeguards'] += ['12.4']
returns.update({'12.1':['11.1'],'12.2':['11.14'],'12.3':['11.15']})
families['Population, risk sets, aggregation and uncertainty delivery'] += ['13.2', '13.3', '13.6']
families['Validation design, diagnostics, provenance and reproducibility'] += ['13.4', '13.5']
families['Nonlinear wave correction, entry and alpha'] += ['13.1']
families['Reference-frailty estimand, scope and benchmark'] += ['13.7', '13.8']
returns.update({'13.1':['12.2']})
families['Population, risk sets, aggregation and uncertainty delivery'] += ['14.1', '14.2', '14.4', '14.5', '14.6', '14.7', '14.8', '14.9']
families['Validation design, diagnostics, provenance and reproducibility'] += ['14.10', '14.11', '14.12']
families['Reference-frailty estimand, scope and benchmark'] += ['14.3']

families['Population, risk sets, aggregation and uncertainty delivery'] += ['15.1','15.2','15.3']
families['Reference-frailty estimand, scope and benchmark'] += ['15.4','15.8','15.9']
families['Validation design, diagnostics, provenance and reproducibility'] += ['15.5','15.6']
families['Nonlinear wave correction, entry and alpha'] += ['15.7']
families['Frailty fitting, information and numerical safeguards'] += ['15.10']
returns.update({'15.1':['14.1','14.2']})

families['Population, risk sets, aggregation and uncertainty delivery'] += ['16.1','16.2','16.3','16.4']
families['Validation design, diagnostics, provenance and reproducibility'] += ['16.5']


families['Population, risk sets, aggregation and uncertainty delivery'] += ['17.1','17.2']
families['Validation design, diagnostics, provenance and reproducibility'] += ['17.3']

families['Population, risk sets, aggregation and uncertainty delivery'] += ['18.1','18.2','18.3','18.4']
families['Nonlinear wave correction, entry and alpha'] += ['18.5']
returns.update({'18.1':['17.1']})

families['Population, risk sets, aggregation and uncertainty delivery'] += ['19.1','19.2','19.3','19.5','19.6','19.7']
families['Validation design, diagnostics, provenance and reproducibility'] += ['19.4']

families['Population, risk sets, aggregation and uncertainty delivery'] += ['20.1','20.2']
families['Frailty fitting, information and numerical safeguards'] += ['20.3']

families['Population, risk sets, aggregation and uncertainty delivery'] += ['21.1','21.3']
families['Validation design, diagnostics, provenance and reproducibility'] += ['21.2','21.4']

early_related={'2.1':['1.5'],'2.2':['1.6'],'2.3':['1.2'],'2.4':['1.2'],'2.5':['1.1','1.4'],'2.6':['1.5'],'2.7':['1.3'],'2.8':['1.5','2.7'],'2.9':['1.3'],'2.10':['1.4'],'3.1':['2.7','1.3'],'3.2':['1.2','2.3'],'3.3':['2.6'],'4.1':['1.2','3.2'],'4.2':['1.2'],'4.3':['1.2'],'4.4':['2.3','3.1'],'4.5':['3.2'],'4.6':['1.3','2.9'],'4.7':['1.2','1.4'],'4.8':['2.6','3.3'],'4.9':['1.1','2.5']}
nodes={}
for d in sorted(ob.iterdir(),key=lambda p:tuple(map(int,p.name.split('.'))) if re.fullmatch(r'\d+\.\d+',p.name) else (999,999)):
 if not re.fullmatch(r'\d+\.\d+',d.name):continue
 p=d/'README.md'
 if not p.exists():continue
 s=p.read_text(encoding='utf8',errors='replace');n=d.name
 title=s.splitlines()[0].lstrip('# ').split(' - Round 5 finding')[0]
 if title==n:
  prose=next((x for x in s.splitlines()[1:] if x.strip() and not x.startswith(('**','#','['))),n)
  title=prose[:125]+('...' if len(prose)>125 else '')
 status=next((line.strip().replace('**','') for line in s.splitlines() if line.startswith('**Status')), 'See objection record; no uniform status line.')
 dispositions=[line.strip().replace('**','') for line in s.splitlines() if re.match(r'\*\*Round \d+ independent disposition',line)]
 disposition=dispositions[-1] if dispositions else None
 related=set(early_related.get(n,[]))
 for line in s.splitlines():
  if line.startswith(('**Related issues','**Related root causes')):related.update(re.findall(r'\b\d+\.\d+\b',line))
 related.discard(n)
 nodes[n]=dict(title=title,status=status,disposition=disposition,returns=returns.get(n,[]),related=sorted(related,key=lambda x:tuple(map(int,x.split('.')))))
assert all(n in nodes for ids in families.values() for n in ids)
assert set(nodes)==set(n for ids in families.values() for n in ids)
def link(n):return f'[{n}](objections/{n}/README.md)'
def links(ids):return ', '.join(link(n) for n in ids if n in nodes) or '-'
def esc(x):return x.replace('|','\\|').replace('\n',' ')
s='# Objection lineage and hierarchy\n\nThis is the cross-round scientific work index, not the lead-author briefing. Each objection keeps its original x.y identifier and evidence folder. A direct return means a later review explicitly challenges a still-unresolved part of an earlier objection; a related link means a shared mechanism or evidence dependency, not that a previously corrected objection is reopened. Root-cause families organize the whole history; objections may have links across families. Implementation and independent closure are different statuses.\n\n## Root-cause hierarchy\n\n'
for name,ids in families.items():
 s+=f'### {name}\n\n'+links(ids)+'\n\n'
s+='## Direct recurrence branches\n\n'
children={}
for n,parents in returns.items():
 for p in parents:children.setdefault(p,[]).append(n)
def branch(n,depth=0):
 return '  '*depth+'- '+link(n)+' - '+nodes[n]['title'].replace(n+' - ','',1)+'\n'+''.join(branch(c,depth+1) for c in children.get(n,[]))
for parent in children:
 if parent not in returns:s+=branch(parent)
s+='\n## Round-by-round register\n'
for rnd in range(1,1+max(int(n.split('.')[0]) for n in nodes)):
 s+=f'\n### Round {rnd}\n\n'
 if rnd>1:s+=f'[Evaluation records](iterative%20review/evaluations/{rnd}/)\n\n'
 s+='| Objection | Title / finding | Direct predecessor | Related objections | Implementation / review status |\n|---|---|---|---|---|\n'
 for n,v in nodes.items():
  if int(n.split('.')[0])!=rnd:continue
  status=v['status']+(' '+v['disposition'] if v['disposition'] else '')
  if len(status)>280:status=status[:277]+'...'
  s+=f"| {link(n)} | {esc(v['title'])} | {links(v['returns'])} | {links(v['related'])} | {esc(status)} |\n"
s+='\n## Maintenance rule\n\nAfter each evaluation, add the new x.y entries, record each direct predecessor where established, cross-link shared root causes without declaring recurrence, and record independent resolution separately from implementation. Update this register when a correction is applied; preserve all prior checkpoints and review artifacts.\n'
(base/'OBJECTION-LINEAGE.md').write_text(s,encoding='utf8')
(base/'iterative review/evaluations/objection-lineage.json').write_text(json.dumps(dict(nodes=nodes,families=families),indent=2,ensure_ascii=False),encoding='utf8')
p=base/'00 START HERE.md';s=p.read_text(encoding='utf8');s=s.replace('The [numbered objection index](README.md) leads to all 19 folders in [objections/](objections/).','The [objection lineage and hierarchy](OBJECTION-LINEAGE.md) tracks every numbered objection, returns, linked root causes and implementation/review status across rounds. The [numbered objection index](README.md) leads to the historical objection folders.');p.write_text(s,encoding='utf8')
p=base/'iterative review/evaluations/README.md';s=p.read_text(encoding='utf8')
if '[cross-round objection lineage]' not in s:
    s+='\nMaintain the [cross-round objection lineage](../../OBJECTION-LINEAGE.md) and its machine-readable companion `objection-lineage.json` after every review and approved correction. Record direct returns separately from related root causes and implementation separately from independent closure.\n';p.write_text(s,encoding='utf8')
print('Indexed',len(nodes),'objections in',len(families),'families')
