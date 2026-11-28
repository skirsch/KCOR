from pathlib import Path
import re,sys
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8').splitlines();b=new.read_text(encoding='utf8').splitlines();s=red.read_text(encoding='utf8')
def mark(n='8.5'):return r'\textsuperscript{\hyperlink{issue-'+n+'}{'+n+'}} '
def strike(x):
    # Preserve math fragments as single balanced tokens, and allow ordinary text to wrap.
    tokens=re.findall(r'\\\(.*?\\\)|\\texttt\{[^{}]*\}|\S+',x)
    return ' '.join(r'\oldword{'+v+'}' for v in tokens)
def change(x,y,n='8.5'):
 global s
 assert x in s,x[:120]
 s=s.replace(x,mark(n)+strike(x)+r' {\color{red} '+y+'}',1)
def diff(i,n='8.5'):
 x,y=a[i].split(),b[i].split();pre=0
 while pre<min(len(x),len(y)) and x[pre]==y[pre]:pre+=1
 suf=0
 while suf<min(len(x)-pre,len(y)-pre) and x[-1-suf]==y[-1-suf]:suf+=1
 oldpart=' '.join(x[pre:len(x)-suf]);newpart=' '.join(y[pre:len(y)-suf])
 change(oldpart,newpart,n)
for i in (229,309,1328):diff(i)
change(r'\(\alpha\) is not separately identifiable from cohort-specific multiplicative intervention effects in minimal aggregated data,',r'a single cross-cohort excess-hazard contrast can confound \(\alpha\) with an unknown intervention multiplier,')
x=r'{\color{red} localizing \(\alpha\) with} this module';assert x in s
s=s.replace(x,mark()+strike(r'localizing \(\alpha\) with this module')+r' {\color{red} the proposed localization procedure}',1)
change(r'Under minimal aggregated data, the NPH exponent \(\alpha\) is not separately identifiable from cohort-specific multiplicative intervention effects if both are left free. Accordingly, when the optional NPH module is used to localize \(\alpha\),',r'The proposed localization procedure does not jointly estimate the NPH exponent \(\alpha\) and cohort-specific intervention multipliers. When this procedure is used,')
x=r'KCOR does not identify that scale from the same aggregated wave-period contrasts used to localize \(\alpha\).'
y=b[696].split(x,1)[1]
assert x in s and y
s=s.replace(x,x+' '+mark()+r'{\color{red} '+y.strip()+'}',1)
change(r'Identification of \(\alpha\) requires sufficient cross-cohort variation',r'In the proposed localization procedure, identification of \(\alpha\) requires sufficient cross-cohort variation')
change(r'When the optional NPH module is used, identifiability of \(\alpha\) is likewise conditional:',r'When the proposed optional NPH localization procedure is used, identifiability of \(\alpha\) is likewise conditional:')
change(r'\(\alpha\) is not separately identifiable from cohort-specific multiplicative effects under minimal data if that scale is not fixed externally',r'the procedure does not jointly identify \(\alpha\) and cohort-specific multiplicative effects if that scale is not fixed externally')
change(r'Under minimal aggregated data, \(\alpha\) is not separately identifiable from cohort-specific multiplicative intervention effects unless',r'The proposed localization procedure requires')
change(r'is fixed and not estimated from the same wave-period contrasts used to localize \(\alpha\)',r', fixed rather than estimated jointly with \(\alpha\) from the wave-period contrasts')
# Remove whitespace before the punctuation introduced by the tail edit.
s=s.replace(r'\textbf{externally supplied VE or other prespecified intervention-effect scale} '+mark()+r'\oldword{is}',r'\textbf{externally supplied VE or other prespecified intervention-effect scale} '+mark()+r'\oldword{is}',1)
diff(2611,'8.4')
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-8.5}{}\textbf{8.5} Qualify external-scale requirement as a property of the proposed alpha-localization procedure, not a universal identification theorem. No estimator change.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
