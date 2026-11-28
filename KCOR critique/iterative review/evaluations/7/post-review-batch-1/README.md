# Round 7 approved straightforward corrections

Candidate m129 retains frozen m128. This batch addresses 7.1, 7.2, 7.4, 7.8 and 7.10. No optimizer scaling or new scientific redesign, no full rebootstrap, no PR update.

[Workbook patch](workbook.patch) | [Paper patch](paper.patch) | [Tests](tests.patch) | [Table 8 point replay](round7_table8_point_check.json) | [Clean paper](KCOR-clean.pdf) | [Redline](KCOR-redline.pdf)

The targeted ordinary workbook, output/bootstrap, previous aggregate and flag regressions passed. The module-off Table 8 points match their archive to machine precision; forced-alpha points differ by less than 0.000001 due to the previous numerical integration change. All 300 S8 archived intervals reproduce from the original draws. The historical generation source files are archived with a manifest; they are not presented as a fresh standalone current simulation.
