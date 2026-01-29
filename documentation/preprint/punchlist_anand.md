**KCOR: A Depletion-Neutralized Framework for Retrospective Cohort**

**Comparison Under Latent Frailty**

+-------------------------------+--------------------------------+-----------------+------------+
| **Article Type:** Methods /   | **Running title**: KCOR under  | **Author**:     | **Word     |
| Statistical method            | selection-induced cohort bias  | Steven T.       | count:**   |
|                               |                                | Kirsch          | 10,909     |
+-----------+---------+---------+----------+----------+----------+--------+--------+------------+
| **Key**   | **Low   |                    | **Medium |                   | **High |            |
|           | Risk**  |                    | Risk**   |                   | Risk** |            |
+===========+=========+=========+==========+==========+==========+========+========+============+

: Layout table to add Sheet Number, Date, Performed by, and Department

+-----------------------------------------------------------------------------------------------------------------------------------+
| **ABSTRACT**                                                                                                                      |
+================+================+:========================+:============================+========================+================+
| ABS-02         | Abstract, Line | \"KCOR is a             | \"Depletion-neutralized\"   | Rephrase to: \"KCOR is | Medium         |
|                | 3              | depletion-neutralized   | is \"branded\" terminology. | a framework that       |                |
|                |                | cohort comparison       | Reviewers prefer            | adjusts for            |                |
|                |                | framework\...\"         | descriptive statistical     | selection-induced      |                |
|                |                |                         | language over new           | frailty by applying a  |                |
|                |                |                         | proprietary-sounding terms. | gamma-frailty          |                |
|                |                |                         |                             | normalization\...\"    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| ABS-03         | Abstract, Line | Mention of \"depletion  | \"Geometry\" is vague. It   | Change to \"depletion  | Low            |
|                | 4              | geometry.\"             | implies a shape without     | parameters\" or \"the  |                |
|                |                |                         | defining the parameters.    | rate of frailty-driven |                |
|                |                |                         |                             | selection.\"           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| ABS-04         | Abstract, Line | Use of \"quiet          | This is a critical          | Add a brief qualifier: | High           |
|                | 4              | periods\" without       | methodological dependency.  | \"\...during           |                |
|                |                | definition.             | If the reviewer doesn\'t    | epidemiologically      |                |
|                |                |                         | understand what makes a     | quiet periods          |                |
|                |                |                         | period \"quiet,\" they will | (intervals of stable   |                |
|                |                |                         | distrust the normalization. | baseline risk)\"       |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| ABS-05         | Abstract, Line | \"\...map observed      | This sounds like \"data     | Specify that the       | Medium         |
|                | 5              | cumulative hazards into | transformation\" without a  | target scale is the    |                |
|                |                | a common comparison     | clear target.               | \"frailty-neutral\" or |                |
|                |                | scale.\"                |                             | \"unselected\"         |                |
|                |                |                         |                             | population baseline.   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| ABS-06         | Abstract, Line | \"Cox proportional      | Reviewers know Cox is       | Frame it as: \"Under   | Medium         |
|                | 7              | hazards regression can  | biased if assumptions are   | conditions of latent   |                |
|                |                | exhibit systematic      | violated. The claim needs   | heterogeneity,         |                |
|                |                | bias\...\"              | to be tied specifically to  | standard models fail   |                |
|                |                |                         | the frailty violation, not  | to satisfy the         |                |
|                |                |                         | the model itself.           | proportionality        |                |
|                |                |                         |                             | assumption, leading    |                |
|                |                |                         |                             | to\...\"               |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| ABS-07         | Abstract, End  | Missing \"Scope\" or    | A reviewer will fear the    | Explicitly state:      | High           |
|                |                | \"What it doesn\'t do\" | author is presenting KCOR   | \"While KCOR           |                |
|                |                | statement.              | as a \"silver bullet\" for  | neutralizes            |                |
|                |                |                         | all observational study     | selection-induced      |                |
|                |                |                         | flaws (like unmeasured      | depletion, it does not |                |
|                |                |                         | confounding).               | account for external   |                |
|                |                |                         |                             | confounding or         |                |
|                |                |                         |                             | exposure               |                |
|                |                |                         |                             | misclassification.\"   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| ABS-08         | Keywords       | Extensive list of       | Oversaturation of keywords  | Trim the list to the 6 | Low            |
|                |                | keywords.               | can look like \"search      | most impactful terms.  |                |
|                |                |                         | engine optimization\"       | Remove redundant       |                |
|                |                |                         | rather than academic        | terms.                 |                |
|                |                |                         | indexing.                   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **1 INTRODUCTION**                                                                                                                |
|                                                                                                                                   |
| **1.1 Retrospective Cohort Comparisons Under Selection**                                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.1-01     | Para 1, Line   | Conventional \"RCT vs.  | This is a standard          | Condense the first     | Low            |
|                | 1-3            | Observational\"         | \"filler\" opening.         | three sentences. Pivot |                |
|                |                | opening.                | Reviewers for high-impact   | quickly to why         |                |
|                |                |                         | journals prefer an          | *standard*             |                |
|                |                |                         | immediate focus on the      | observational methods  |                |
|                |                |                         | specific methodological gap | fail specifically in   |                |
|                |                |                         | (time-varying selection).   | the presence of        |                |
|                |                |                         |                             | depletion.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.1-02     | Para 2, Line 2 | Mention of \"fixed      | Technical ambiguity. Many   | Clarify if the         | High           |
|                |                | cohort.\"               | registry studies use        | \"fixed\" nature is a  |                |
|                |                |                         | dynamic/open cohorts. If    | requirement of the     |                |
|                |                |                         | KCOR assumes a closed       | KCOR estimator or a    |                |
|                |                |                         | cohort (enrolled at t=0),   | simplification for     |                |
|                |                |                         | this must be explicitly     | this text.             |                |
|                |                |                         | stated.                     |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.1-03     | Para 2, Line 4 | Claim that mortality is | This is a \"strong-form\"   | Soften the language:   | Medium         |
|                |                | \"free from             | claim that invites          | \"Mortality\... is     |                |
|                |                | outcome-dependent       | pushback. Registry death    | generally less         |                |
|                |                | ascertainment biases.\" | data often suffers from     | susceptible to         |                |
|                |                |                         | reporting lags, \"death     | ascertainment bias     |                |
|                |                |                         | sweeps,\" and               | than self-reported or  |                |
|                |                |                         | misclassification of cause. | clinically coded       |                |
|                |                |                         |                             | endpoints.\"           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.1-04     | Para 3, Line 2 | Lack of citation for    | This is a major theoretical | Add a citation         | Low            |
|                |                | \"non-exchangeability   | pivot point. It needs to be | regarding the          |                |
|                |                | evolve differently.\"   | anchored in existing        | distinction between    |                |
|                |                |                         | literature (e.g., Hernán on | baseline confounding   |                |
|                |                |                         | selection bias/colliders).  | and selection bias     |                |
|                |                |                         |                             | over follow-up.        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.1-05     | Para 3, Line 3 | \"factors that          | This describes confounding. | Explicitly distinguish | Medium         |
|                |                | influence treatment     | KCOR focuses on             | between baseline       |                |
|                |                | uptake also influence   | frailty-induced selection   | confounding (handled   |                |
|                |                | outcome risk.\"         | (a form of collider bias    | by IPTW/Matching) and  |                |
|                |                |                         | over time). Mixing these    | the *evolution* of     |                |
|                |                |                         | terms can confuse the       | cohort composition     |                |
|                |                |                         | reader.                     | (handled by KCOR).     |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.1-06     | Para 4, Line 1 | \"This manuscript is a  | While transparent, this     | Move the               | Low            |
|                |                | methods paper.\"        | feels like a defensive      | \"methods-only\"       |                |
|                |                |                         | disclaimer. It can signal   | stance to a dedicated  |                |
|                |                |                         | to a reviewer that the      | \"Scope of Work\"      |                |
|                |                |                         | author is \"shielding\" the | sub-section or         |                |
|                |                |                         | method from real-world      | integrate it into      |                |
|                |                |                         | scrutiny.                   | \"Contribution\"       |                |
|                |                |                         |                             | section                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.1-07     | Para 4, Line 1 | \"Real-world registry   | Reviewers often find        | Ensure the text        | Medium         |
|                |                | data are used solely to | \"demonstration only\" data | mentions *which*       |                |
|                |                | demonstrate\...\"       | frustrating if the data     | registry (even if      |                |
|                |                |                         | source isn\'t described     | anonymized) and the    |                |
|                |                |                         | with enough rigor to be     | nature of the          |                |
|                |                |                         | reproducible.               | \"realistic\" noise    |                |
|                |                |                         |                             | injected into it.      |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **1.2 Curvature (Shape) is the Hard Part: Non-Proportional Hazards from Frailty Depletion**                                       |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-01     | Page 1,        | Section title:          | Academic sub-headings       | You may rename it to:  | Low            |
|                | Section        | \"Curvature (shape) is  | should be descriptive and   | \"Selection-Induced    |                |
|                | Heading 1.2    | the hard part\"         | formal. \"The hard part\"   | Non-Proportionality    |                |
|                |                |                         | is too colloquial for a     | and Cumulative Hazard  |                |
|                |                |                         | methods paper.              | Concavity.\"           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-02     | Para 2, Line 3 | \"\...generally         | \"Negative concavity\" is a | Use \"downward         | Medium         |
|                |                | produces negative       | double negative/redundant.  | concavity\" or         |                |
|                |                | concavity\...\"         | Concavity itself implies a  | \"convexity of the     |                |
|                |                |                         | downward bend in this       | survival function\"    |                |
|                |                |                         | context.                    | (the latter is the     |                |
|                |                |                         |                             | standard terminology   |                |
|                |                |                         |                             | in frailty theory).    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-03     | Para 3 and 4   | Critique of Cox PH as a | Reviewers may argue that    | Specify:               | Medium         |
|                | of Section 1.2 | \"failure mode.\"       | \"Cox with time-varying     | \"\...standard         |                |
|                |                |                         | coefficients\" or           | fixed-coefficient Cox  |                |
|                |                |                         | \"stratified Cox\" can      | PH models fail to      |                |
|                |                |                         | handle this. You must be    | capture the evolution  |                |
|                |                |                         | specific about why standard | of the hazard ratio    |                |
|                |                |                         | Cox fails.                  | under differential     |                |
|                |                |                         |                             | depletion.\"           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-04     | Para 4, Line 4 | \"Approximate           | This is a very strong and   | Add: \"\...assuming a  | High           |
|                |                | linearity\... serves as | useful claim, but it needs  | stable baseline hazard |                |
|                |                | an internal             | a caveat. Linearity only    | during the diagnostic  |                |
|                |                | diagnostic.\"           | indicates \"selection       | window.\"              |                |
|                |                |                         | removal\" if the underlying |                        |                |
|                |                |                         | hazard is truly constant.   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-05     | Para 5 of      | Mention of COVID-19     | COVID data is highly        | Reinforce that the     | Low            |
|                | Section 1.2    | period.                 | \"noisy.\" A reviewer might | logic applies to any   |                |
|                |                |                         | worry that the method is    | cohort with latent     |                |
|                |                |                         | over-fitted to the specific | heterogeneity (e.g.,   |                |
|                |                |                         | volatility of 2021-2022.    | cancer registries or   |                |
|                |                |                         |                             | aging studies).        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-06     | Para 6 of      | \"Static HVE\" defined  | This is a novel             | Explicitly state: \"We | Medium         |
|                | Section 1.2    | as \"differing          | re-definition of Healthy    | propose a mechanistic  |                |
|                |                | depletion curvature.\"  | Vaccinee Effect. Usually,   | interpretation of      |                |
|                |                |                         | HVE is seen as simple       | Static HVE as a        |                |
|                |                |                         | baseline confounding.       | difference in latent   |                |
|                |                |                         |                             | frailty                |                |
|                |                |                         |                             | distributions\...\"    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-07     | Para 6 of      | \"Dynamic HVE\...       | If the window is too long,  | Mention that §2.7      | High           |
|                | Section 1.2    | addressed by            | you lose data; if too       | provides a sensitivity |                |
|                |                | prespecifying a         | short, you keep bias.       | analysis or objective  |                |
|                |                | skip/stabilization      | Reviewers will want to know | criteria for this      |                |
|                |                | window.\"               | how the \"4-week\" (or      | window selection.      |                |
|                |                |                         | similar) choice is          |                        |                |
|                |                |                         | justified.                  |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.2-08     | Para 6 of      | Lack of explicit        | You are describing the      | Add a citation to      | Medium         |
|                | Section 1.2    | mention of \"The        | classic \"Vaupel\'s         | Vaupel et al. (1979)   |                |
|                |                | Frailty Effect.\"       | Paradox\" (pop. hazard      | or Hougaard (1984) to  |                |
|                |                |                         | declines while individual   | demonstrate the        |                |
|                |                |                         | hazard is constant).        | method\'s roots in     |                |
|                |                |                         |                             | established theory.    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **1.3 Related Work (Brief Positioning)**                                                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.3-01     | Page 3,        | \"Brief positioning\"   | Section headers should not  | Remove \"(brief        | Low            |
|                | Section 1.3    | in parentheses.         | contain meta-commentary     | positioning)\" from    |                |
|                | header         |                         | about their own length.     | the header.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.3-02     | Page 3, Line 2 | Citation \[1\] for      | Gamma frailty is a massive  | Cite the \"big         | Medium         |
|                |                | gamma frailty.          | field. Citing one source is | three\": Vaupel        |                |
|                |                |                         | insufficient and looks like | (1979), Hougaard       |                |
|                |                |                         | a shallow lit review.       | (1984), and Wienke     |                |
|                |                |                         |                             | (2010).                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.3-03     | Page 3, Line 6 | Contrast with           | You claim these require     | Clarify that KCOR      | Medium         |
|                |                | MSM/IPW/g-methods.      | \"richer longitudinal       | targets unmeasured     |                |
|                |                |                         | covariates.\" A reviewer    | heterogeneity that     |                |
|                |                |                         | may argue that KCOR also    | standard IPW cannot    |                |
|                |                |                         | requires \"rich\" data (the | see, even with rich    |                |
|                |                |                         | quiet period).              | covariates.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **1.4 Evidence from the Literature: Residual Confounding Despite Meticulous Matching**                                            |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.4-01     | Page 3, Line 3 | Citation \[10, 11\] for | These are your \"motivating | Name the               | Medium         |
|                | of this        | residual HVE.           | studies.\" If the reviewer  | studies/authors in the |                |
|                | section        |                         | has to flip to the end to   | text (e.g., \"As seen  |                |
|                |                |                         | see who they are, you lose  | in the Denmark/Qatar   |                |
|                |                |                         | momentum.                   | studies \[10,          |                |
|                |                |                         |                             | 11\]\...\") to ground  |                |
|                |                |                         |                             | the argument.          |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.4-02     | Page 3, Line   | \"Consistent with       | This is an interpretation,  | Use slightly more      | Low            |
|                | 2-3 of this    | selection and depletion | not a fact. You are         | cautious language:     |                |
|                | section        | dynamics\...\"          | attributing their residual  | \"\...which may be     |                |
|                |                |                         | bias to your specific       | explained by selection |                |
|                |                |                         | mechanism.                  | and depletion          |                |
|                |                |                         |                             | dynamics\...\"         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **1.5 Contribution of this Work**                                                                                                 |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.5-01     | Page 3, Line 1 | Contribution (i): \"it  | Frailty-induced             | Change \"formalizes\"  | Medium         |
|                | of this        | formalizes\...\"        | non-proportionality is      | to \"characterizes\"   |                |
|                | section        |                         | already formalized in the   | or \"applies a         |                |
|                |                |                         | literature.                 | formalization of\...\" |                |
|                |                |                         |                             | to avoid over-claiming |                |
|                |                |                         |                             | novelty in basic       |                |
|                |                |                         |                             | theory.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.5-02     | Page 4, Line   | Contribution (iii):     | This is a very strong part  | Mention that these     | Low            |
|                | 4-5 of this    | \"synthetic and         | of the paper. It should be  | controls specifically  |                |
|                | section        | empirical controls.\"   | emphasized more.            | \"stress-test\" the    |                |
|                |                |                         |                             | depletion assumption.  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.5-03     | Para 2, Line 1 | \"A central implication | Major Statistical Concern.  | If you haven\'t        | High           |
|                | of this        | is identifiability.\"   | \"Identifiability\" has a   | provided a formal      |                |
|                | section        |                         | strict mathematical         | proof, change to: \"A  |                |
|                |                |                         | definition. If you haven\'t | central requirement    |                |
|                |                |                         | proved it, don\'t use the   | for parameter          |                |
|                |                |                         | word.                       | estimation is the      |                |
|                |                |                         |                             | existence of\...\"     |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.5-04     | Para 2, Line 2 | \"\...indicate          | This is the \"Aha!\" moment | This is well-written   | Low            |
|                | of this        | depletion geometry has  | of the paper. It needs to   | but consider a visual  |                |
|                | section        | been\... removed,       | be very clear.              | aid (DAG or Plot) here |                |
|                |                | rather than absorbed    |                             | to show the            |                |
|                |                | into a time-varying     |                             | \"absorption\" vs.     |                |
|                |                | effect.\"               |                             | \"removal\"            |                |
|                |                |                         |                             | distinction.           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.5-05     | Para 3 of this | \"\...position KCOR\... | This is your \"Product      | Keep this phrasing; it | Low            |
|                | section        | as a prerequisite       | Positioning.\" It's clever  | reduces the \"threat\" |                |
|                |                | normalization step.\"   | because it bypasses the     | to established methods |                |
|                |                |                         | \"Is KCOR better than       | and makes KCOR look    |                |
|                |                |                         | Cox?\" argument.            | like a \"utility\"     |                |
|                |                |                         |                             | tool.                  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **1.6 Target Estimand and Scope (Non-Causal)**                                                                                    |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.6-01     | Page 3, Box 2  | Estimand defined as     | hazards are difficult to    | Explicitly contrast    | Medium         |
|                |                | {H}\_{0,A}(t) /         | interpret because they      | this with a            |                |
|                |                | {H}\_{0,B}(t).          | \"carry the history\" of    | \"Cumulative Hazard    |                |
|                |                |                         | the whole period. They are  | Ratio\" from the       |                |
|                |                |                         | not hazard ratios.          | literature and explain |                |
|                |                |                         |                             | why the *neutralized*  |                |
|                |                |                         |                             | version is superior.   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.6-02     | Page 3, Box 2, | \"KCOR is not a causal  | While humble, this creates  | Clarify that KCOR      | Medium         |
|                | Line 12        | effect estimator.\"     | a \"utility gap.\" If a     | provides a descriptive |                |
|                |                |                         | reviewer asks, \"If I       | structural adjustment  |                |
|                |                |                         | can\'t use this to decide   | that is a necessary    |                |
|                |                |                         | if a drug works, why am I   | precursor to any       |                |
|                |                |                         | reading this?\", you need a | causal inquiry in      |                |
|                |                |                         | stronger answer.            | heterogeneous cohorts. |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.6-03     | Page 3, Box 2, | Assumption: \"Fixed     | In registry data, \"fixed\" | Specify that the       | High           |
|                | Line 2         | enrollment cohorts.\"   | usually means \"no new      | cohorts are \"fixed at |                |
|                |                |                         | entries,\" but what about   | t=0 and subject to     |                |
|                |                |                         | censoring? If people leave  | non-informative        |                |
|                |                |                         | the cohort (migration/loss  | censoring,\" or        |                |
|                |                |                         | to follow-up), the          | explain how KCOR       |                |
|                |                |                         | depletion math changes.     | handles administrative |                |
|                |                |                         |                             | censoring.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.6-04     | Page 3, Box 2, | Assumption: \"Shared    | This is your \"Parallel     | Explicitly state that  | High           |
|                | Line 14        | external hazard         | Trends\" assumption. If     | cohorts must be        |                |
|                |                | environment.\"          | Cohort A lives in a         | \"subject to the same  |                |
|                |                |                         | different city than Cohort  | background intensity   |                |
|                |                |                         | B, the \"quiet period\"     | of the event           |                |
|                |                |                         | normalization fails.        | process.\"             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| You need to be careful with INT-1.6-04 (Shared Environment). In the context of the COVID-19 examples you\'ve hinted at, a         |
| reviewer might argue that vaccinated and unvaccinated people have different \"behavioral hazards\" (e.g., masking, social         |
| distancing) that aren\'t \"shared.\" You should briefly acknowledge that the \"Shared Environment\" assumption refers to the      |
| external risk (e.g., community transmission levels), not necessarily individual behavior.                                         |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.6-05     | Page 3, Last   | \"If diagnostics        | A reviewer may want to know | Briefly mention that   | Low            |
|                | Line           | indicate                | the threshold. What         | §3 or §5 defines the   |                |
|                |                | non-identifiability\... | constitutes \"failure\"?    | specific diagnostic    |                |
|                |                | KCOR is not reported.\" |                             | thresholds (e.g., R\^2 |                |
|                |                |                         |                             | of the fit or Wald     |                |
|                |                |                         |                             | test on parameters).   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.6-06     | Page 3, Box 2, | Use of \"Depletion      | As noted in the Abstract,   | Change to \"depletion  | Low            |
|                | Line 6         | Geometry\" again.       | \"geometry\" is a bit too   | parameters\" or        |                |
|                |                |                         | abstract here.              | \"latent frailty       |                |
|                |                |                         |                             | parameters\" to ground |                |
|                |                |                         |                             | it in statistical      |                |
|                |                |                         |                             | theory.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.6-07     | Page 3, Box 2, | \"KCOR is not an        | This is a vital             | Add one sentence       | Medium         |
|                | Line 11        | instantaneous hazard    | distinction. Many readers   | explaining that KCOR   |                |
|                |                | ratio.\"                | will see a ratio and think  | is more akin to a      |                |
|                |                |                         | \"Hazard Ratio.\"           | \"Relative Risk of     |                |
|                |                |                         |                             | having experienced the |                |
|                |                |                         |                             | event by time t,\"     |                |
|                |                |                         |                             | after adjusting for    |                |
|                |                |                         |                             | selection.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **1.7 Paper Organization and Supporting Information (SI)**                                                                        |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.7-01     | Page 4, Line 1 | \"\...provides a        | \"Canonical\" implies this  | Replace \"canonical\"  | Low            |
|                | of Section 1.7 | canonical demonstration | is the definitive,          | with \"illustrative\"  |                |
|                |                | of Cox bias\...\"       | universally accepted        | or \"representative.\" |                |
|                |                |                         | example. This may come off  |                        |                |
|                |                |                         | as overconfident to a       |                        |                |
|                |                |                         | statistician.               |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.7-02     | Page 4, Line 2 | Mention of a \"stress   | \"Stress test\" is common   | Change to              | Low            |
|                | of Section 1.7 | test.\"                 | in engineering or finance   | \"sensitivity analysis |                |
|                |                |                         | but less formal in          | under extreme          |                |
|                |                |                         | biostatistics.              | heterogeneity\" or     |                |
|                |                |                         |                             | \"boundary condition   |                |
|                |                |                         |                             | analysis.\"            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.7-03     | Page 4, Line   | \"Additional            | Major Structural Concern.   | Move at least one key  | High           |
|                | 2-3 of Section | validations---including | Reviewers usually insist    | positive control       |                |
|                | 1.7            | positive controls\...   | that Positive Controls      | figure/summary from    |                |
|                |                | provided in the SI.\"   | (proving the method can     | the SI to the main     |                |
|                |                |                         | actually detect an effect   | Results section.       |                |
|                |                |                         | when one exists) be in the  |                        |                |
|                |                |                         | main text.                  |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.7-04     | Page 4, Line 3 | \"Empirical             | This is a very strong,      | Give this a brief      | Medium         |
|                | of Section 1.7 | registry-based nulls.\" | high-value component of     | \"half-sentence\" of   |                |
|                |                |                         | your validation. It is      | explanation:           |                |
|                |                |                         | currently buried in a list  | \"\...including        |                |
|                |                |                         | of SI contents.             | empirical nulls        |                |
|                |                |                         |                             | derived from           |                |
|                |                |                         |                             | historical registry    |                |
|                |                |                         |                             | data to establish      |                |
|                |                |                         |                             | baseline estimator     |                |
|                |                |                         |                             | variance.\"            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.7-05     | Page 4, Line 4 | Detailed section/table  | While precise, these        | Keep the references    | Low            |
|                | of Section 1.7 | references (e.g.,       | references often break      | but ensure they are    |                |
|                |                | S4.2-S4.3).             | during the final layout or  | hyperlinked in the PDF |                |
|                |                |                         | revision process.           | to make the            |                |
|                |                |                         |                             | reviewer\'s life       |                |
|                |                |                         |                             | easier.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| INT-1.7-06     | Page 4,        | Lack of a \"Data/Code   | For a methods paper, the    | Add a brief sentence:  | Medium         |
|                | General        | Availability\" mention. | \"organization\" section is | \"All code for the     |                |
|                |                |                         | a good place to signal that | KCOR estimator and     |                |
|                |                |                         | the code is open-source.    | simulation environment |                |
|                |                |                         |                             | is available at        |                |
|                |                |                         |                             | \[Link/Repository\].\" |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2. METHODS**                                                                                                                    |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.0-01     | Page 4, Line 1 | Repeating the           | You already stated this in  | Delete or condense to: | Low            |
|                | of this        | \"Mortality is the      | Section 1.1. Redundancy     | \"As noted, mortality  |                |
|                | section        | example\"               | suggests the paper hasn\'t  | serves as the primary  |                |
|                |                | justification.          | been tightly edited.        | endpoint for           |                |
|                |                |                         |                             | methodological         |                |
|                |                |                         |                             | exposition.\"          |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.0-02     | Page 4, Line 3 | Reference to \"Table    | Reviewers hate searching    | Ensure Table 4 is on   | Low            |
|                | of this        | 4\" for notation.       | for tables. If the notation | the same page or the   |                |
|                | section        |                         | is complex, a \"Notation    | very next page as this |                |
|                |                |                         | Box\" or keeping it inline  | mention.               |                |
|                |                |                         | is better.                  |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.1 Conceptual Framework and Estimand**                                                                                         |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-01     | Page 4, Line   | Distinction between     | These are informal terms. A | Define \"Level\" as    | Medium         |
|                | 2-5 of this    | \"Level\" and           | reviewer wants to see these | the baseline hazard    |                |
|                | section        | \"Curvature\"           | defined as h(t) (hazard)    | ratio and              |                |
|                |                | differences.            | vs. h\'(t) (rate of         | \"Curvature\" as the   |                |
|                |                |                         | change).                    | second derivative of   |                |
|                |                |                         |                             | the cumulative hazard  |                |
|                |                |                         |                             | {d\^2} / {dt\^2} H(t)  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-02     | Page 4, Line 6 | \"This framework        | Clarity Check. Does KCOR    | Clarify that KCOR      | Medium         |
|                | of this        | targets the second      | ignore level differences,   | neutralizes curvature  |                |
|                | section        | failure mode.\"         | or does it adjust for       | to enable a valid      |                |
|                |                |                         | curvature so that level     | comparison of          |                |
|                |                |                         | differences can be seen?    | levels/contrasts.      |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-03     | Page 4, Line 8 | \"\...concavity in      | This is a very strong and   | Refer the reader to a  | Low            |
|                | of this        | cumulative-hazard       | important point. It         | specific figure (e.g., |                |
|                | section        | space\... even under a  | deserves a visual aid.      | Fig 1) that            |                |
|                |                | true null.\"            |                             | illustrates this       |                |
|                |                |                         |                             | \"spurious             |                |
|                |                |                         |                             | concavity.\"           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-04     | Page 4, Line   | Strategy Step 1:        | This is the \"Achilles\'    | Add a parenthetical:   | High           |
|                | 13 of this     | \"Estimating\...        | heel.\" If there is no      | \"(See §2.5 for        |                |
|                | section        | during\... quiet        | quiet period, the method    | objective criteria for |                |
|                |                | periods.\"              | fails.                      | \'quiet period\'       |                |
|                |                |                         |                             | selection).\"          |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-05     | Page 4, Line   | \"Discrete weekly time  | Math-Logic Gap. Depletion   | Briefly justify why    | Medium         |
|                | 18 of this     | bins; continuous-time   | is a continuous process. If | \"weekly\" is          |                |
|                | section        | notation is used for    | your bins are too wide, you | sufficient (e.g.,      |                |
|                |                | convenience.\"          | miss the depletion rate     | \"given the rate of    |                |
|                |                |                         | (the \"Vaupel effect\").    | event accumulation\")  |                |
|                |                |                         |                             | to avoid               |                |
|                |                |                         |                             | discretization bias    |                |
|                |                |                         |                             | critiques.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-06     | Page 4, Line 2 | \"observed cohort-level | Does \"fixed\" mean you     | Define \"fixed risk    | High           |
|                | of the         | cumulative hazard,      | ignore new entries (closed  | set\" precisely: \"A   |                |
|                | notation       | computed from fixed     | cohort) or that you use a   | cohort defined by      |                |
|                | preview        | risk sets.\"            | life-table approach?        | enrolment at t=0 with  |                |
|                |                |                         |                             | no subsequent          |                |
|                |                |                         |                             | entries.\"             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-07     | Page 4, Line 4 | \"latent multiplicative | The assumption that theta   | Briefly state *why* we | Medium         |
|                | of the         | frailty term z, with    | differs *by cohort* is the  | expect theta to differ |                |
|                | notation       | cohort-specific         | core of your model.         | (e.g., \"due to        |                |
|                | preview        | variance theta_d.\"     |                             | differential selection |                |
|                |                |                         |                             | pressure during        |                |
|                |                |                         |                             | enrolment\").          |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1-08     | Page 4, Line 5 | \"Full notation is      | (Self-Correction) If Table  | Ensure theta is        | Low            |
|                | of the         | summarized in Table     | 4 is the anchor, make sure  | defined clearly as the |                |
|                | notation       | 4.\"                    | it includes the units for   | variance of the Gamma  |                |
|                | preview        |                         | theta                       | distribution           |                |
|                |                |                         |                             | (E\[z\]=1,             |                |
|                |                |                         |                             | Var\[z\]=theta).       |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.1.1 Target Estimand**                                                                                                         |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.1-01   | Page 5,        | Equation (1): Ratio of  | Ratios of cumulative        | Acknowledge that KCOR  | High           |
|                | Equation 1     | cumulative hazards.     | hazards {H(t)} / {H(t)} are | is a \"running total\" |                |
|                |                |                         | \"history-dependent.\" A    | of the neutralized     |                |
|                |                |                         | single outlier event at t=2 | hazard and explain why |                |
|                |                |                         | will shift the KCOR value   | this is more robust    |                |
|                |                |                         | for all subsequent time     | than (noisy) weekly    |                |
|                |                |                         | points.                     | hazard ratios.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.1-03   | Page 5,        | Introduction of         | Anchoring (dividing by      | Provide a technical    | Medium         |
|                | follow-up      | \"Anchored KCOR.\"      | KCOR(t_0)) can mask         | rationale for the      |                |
|                | Equation to    |                         | early-period bias or        | \"4-week\" example. Is |                |
|                | Eq. 1          |                         | amplify noise if t_0 is     | this based on the      |                |
|                |                |                         | poorly chosen.              | \"stabilization        |                |
|                |                |                         |                             | window\" in §2.7? Link |                |
|                |                |                         |                             | the two sections.      |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.1-05   | Page 5, Last   | \"prespecified t_0      | \"Prespecified\" is good,   | MET-2.1.1-05           | Low            |
|                | Line of this   | (e.g., 4 weeks).\"      | but \"e.g.\" is vague.      |                        |                |
|                | section        |                         | Reviewers want to know if   |                        |                |
|                |                |                         | there is a principled way   |                        |                |
|                |                |                         | to choose t_0 (e.g., after  |                        |                |
|                |                |                         | the Dynamic HVE period).    |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.1.2 Identification Versus Diagnostics**                                                                                       |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.2-01   | Page 5, Line 1 | Reference to Box 2      | The Methods section should  | Briefly restate that   | Medium         |
|                | of this        | (§1.6) for scope.       | be mathematically           | tilde{H}\_{0,d}(t)     |                |
|                | section        |                         | self-contained. Forcing a   | represents the         |                |
|                |                |                         | reviewer to flip back 5     | \"intrinsic\" hazard   |                |
|                |                |                         | pages to find the           | of the population if   |                |
|                |                |                         | \"meaning\" of a variable   | no selection           |                |
|                |                |                         | disrupts technical          | (depletion) had        |                |
|                |                |                         | evaluation.                 | occurred.              |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.2-02   | Page 5, Line 6 | Notation:               | In Cox models, \"baseline   | Consider \"latent      | Medium         |
|                | of this        | tilde{H}\_{0,d}(t)      | hazard\" is the hazard when | cumulative hazard\" or |                |
|                | section        | labeled as              | covariates are zero. In     | \"intrinsic cumulative |                |
|                |                | \"baseline.\"           | KCOR, it\'s the             | hazard\" to avoid      |                |
|                |                |                         | \"selection-neutral\"       | confusion with         |                |
|                |                |                         | hazard. This collision of   | standard Cox baseline  |                |
|                |                |                         | terminology is a potential  | hazards.               |                |
|                |                |                         | trap.                       |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.1.3 KCOR Assumptions and Diagnostics**                                                                                        |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.3-01   | Page 5, Line   | Assumption 1:           | Registry data is rarely     | Explicitly add:        | High           |
|                | 3-4 of this    | \"followed forward      | \"fixed.\" People migrate,  | \"\...and subject only |                |
|                | section        | without dynamic entry   | lose insurance coverage, or | to non-informative     |                |
|                |                | or rebalancing.\"       | die of competing risks.     | administrative         |                |
|                |                |                         | This is censoring, and it   | censoring.\"           |                |
|                |                |                         | must be addressed.          |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.3-02   | Page 5, Line   | Assumption 2:           | This assumes the frailty z  | Clarify if the frailty | Medium         |
|                | 5-6 of this    | \"Multiplicative latent | acts as a constant          | distribution is        |                |
|                | section        | frailty.\"              | multiplier on the hazard.   | assumed to be static   |                |
|                |                |                         | If the intervention         | post-enrollment or if  |                |
|                |                |                         | *changes* the frailty       | the intervention can   |                |
|                |                |                         | distribution (e.g., a       | interact with z.       |                |
|                |                |                         | vaccine that protects the   |                        |                |
|                |                |                         | frail more than the         |                        |                |
|                |                |                         | healthy), this breaks.      |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.3-03   | Page 5, Line   | Assumption 3:           | \"Minimal\" is subjective.  | Define \"minimal       | High           |
|                | 7-8 of this    | \"Quiet-window          | Seasonality in mortality    | shocks\" as: \"The     |                |
|                | section        | stability\... external  | (winter peaks) occurs even  | baseline hazard h_0(t) |                |
|                |                | shocks\... are          | in \"quiet\" years. A       | is assumed to be       |                |
|                |                | minimal.\"              | reviewer will fear seasonal | constant or slowly     |                |
|                |                |                         | trends are being mistaken   | varying relative to    |                |
|                |                |                         | for depletion.              | the rate of            |                |
|                |                |                         |                             | depletion.\"           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.3-04   | Page 5, Line   | Assumption 4:           | In infectious disease (your | Acknowledge that while | Medium         |
|                | 10-11 of this  | \"Independence across   | COVID example), cohorts are | cohorts are analyzed   |                |
|                | section        | strata.\"               | coupled via transmission.   | independently, they    |                |
|                |                |                         | If one cohort gets          | must share the same    |                |
|                |                |                         | infected, it changes the    | external exposure      |                |
|                |                |                         | hazard for the other.       | environment.           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.3-05   | Page 5, Line   | Assumption 5:           | This is vague. If you have  | Quantify               | Medium         |
|                | 12-13 of this  | \"Sufficient event-time | only 3 data points in a     | \"sufficient\": e.g.,  |                |
|                | section        | resolution.\"           | \"quiet window,\" you       | \"The number of time   |                |
|                |                |                         | can\'t fit a depletion      | bins in the quiet      |                |
|                |                |                         | curve reliably.             | window must exceed the |                |
|                |                |                         |                             | number of estimated    |                |
|                |                |                         |                             | depletion              |                |
|                |                |                         |                             | parameters.\"          |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.3-06   | Page 5, Line   | \"Violations are        | This is a great diagnostic, | Suggest a specific     | Low            |
|                | 14-15 (last    | expected to manifest as | but it needs a Null         | diagnostic test, such  |                |
|                | two lines)     | residual                | Hypothesis. How do we know  | as a \"Runs test\" or  |                |
|                |                | curvature\...\"         | if curvature is \"residual  | \"R-squared            |                |
|                |                |                         | noise\" or \"model          | threshold\" for the    |                |
|                |                |                         | failure\"?                  | post-normalization     |                |
|                |                |                         |                             | linear fit.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.1.3-07   | Page 5,        | Lack of justification   | Why Gamma and not Inverse   | Briefly note (or point | Medium         |
|                | General        | for Gamma.              | Gaussian or Log-Normal?     | to §2.4) that          |                |
|                | Comment        |                         |                             | Gamma-frailty is used  |                |
|                |                |                         |                             | for its mathematical   |                |
|                |                |                         |                             | tractability           |                |
|                |                |                         |                             | (closed-form           |                |
|                |                |                         |                             | inversion).            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.2 Cohort Construction**                                                                                                       |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.2-01     | Page 5, Line 1 | \"Required inputs are   | In the eyes of a reviewer,  | Pivot the focus:       | Medium         |
|                | of this        | minimal\...\"           | \"minimal inputs\" is often | \"KCOR is optimized    |                |
|                | section        |                         | code for \"uncontrolled for | for high-volume        |                |
|                |                |                         | confounding.\"              | registry data where    |                |
|                |                |                         |                             | covariate depth is     |                |
|                |                |                         |                             | limited but population |                |
|                |                |                         |                             | coverage is high.\"    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.2-02     | Page 5, Last   | \"No censoring is       | Major Structural Concern.   | Acknowledge that this  | High           |
|                | Line           | applied (other than     | In registry data,           | assumes                |                |
|                |                | administrative\...)\"   | \"out-migration\" (people   | \"non-informative loss |                |
|                |                |                         | moving away) is a form of   | to follow-up\" or that |                |
|                |                |                         | censoring. If you ignore    | the rate of migration  |                |
|                |                |                         | it, your risk set is        | is negligible relative |                |
|                |                |                         | artificially inflated.      | to the mortality rate. |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.2-03     | Page 5, Line 4 | \"Cohorts are           | If the enrollment interval  | Specify if \"event     | High           |
|                | of this        | assigned\... at the     | is long (e.g., 6 months),   | time t\" is measured   |                |
|                | section        | start of the enrollment | the \"start\" is different  | from cohort start      |                |
|                |                | interval.\"             | for everyone.               | (calendar time) or     |                |
|                |                |                         |                             | individual enrolment.  |                |
|                |                |                         |                             | This is a vital        |                |
|                |                |                         |                             | distinction for        |                |
|                |                |                         |                             | depletion math.        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.2-04     | Page 6, Line 4 | \"Allowing transitions  | This is a brilliant         | Ensure this logic is   | Low            |
|                |                | would\... \[change\]    | technical defense of a      | highlighted. It        |                |
|                |                | frailty composition\... | \"fixed cohort\" design. It | explains why KCOR is   |                |
|                |                | in unpredictable        | prevents the reviewer from  | an                     |                |
|                |                | ways.\"                 | asking for a \"Time-Varying | \"Intention-to-Treat\" |                |
|                |                |                         | Cox\" model.                | (ITT) style analysis   |                |
|                |                |                         |                             | of a starting          |                |
|                |                |                         |                             | population.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.2-05     | Page 6, Line 6 | Mention of              | Using \"artifacts\" sounds  | Use the standard term  | Low            |
|                |                | \"immortal-time         | slightly defensive.         | \"immortal time bias\" |                |
|                |                | artifacts.\"            |                             | and explain exactly    |                |
|                |                |                         |                             | how the                |                |
|                |                |                         |                             | fixed-enrollment       |                |
|                |                |                         |                             | window eliminates it.  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.2-06     | Page 6 Last    | \"not framed as a       | If a reviewer is interested | Clarify that all-cause | Medium         |
|                | Line of this   | cause-specific          | in a specific cause of      | mortality is used      |                |
|                | section        | competing-risks         | death, they will argue that | because it provides    |                |
|                |                | analysis.\"             | all-cause mortality         | the clearest signal of |                |
|                |                |                         | \"dilutes\" the signal.     | selection-induced      |                |
|                |                |                         |                             | depletion without the  |                |
|                |                |                         |                             | bias of miscoded       |                |
|                |                |                         |                             | causes.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.2-07     | Page 6, Line   | \"Extensions\... are    | Reviewers often skip the    | Rephrase: \"The        | Medium         |
|                | 7-8            | treated as sensitivity  | Supplement. If §5.2         | robustness of the      |                |
|                |                | analyses (§5.2).\"      | contains the \"real world\" | fixed-cohort           |                |
|                |                |                         | validation (with            | assumption to          |                |
|                |                |                         | censoring), it should be    | realistic levels of    |                |
|                |                |                         | mentioned as a \"robustness | censoring is evaluated |                |
|                |                |                         | check\" here.               | in §5.2.\"             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.3 Hazard Estimation and Cumulative Hazards in Discrete Time**                                                                 |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.3-01     | Page 6, Line 6 | \"there is no loss to   | Major Technical Critique.   | Explicitly state that  | High           |
|                | of this        | follow-up; therefore    | In all-cause mortality      | N_d(t) should be       |                |
|                | section        | N_d(t) is the risk      | studies using registry      | adjusted for known     |                |
|                |                | set\...\"               | data, out-migration and     | censoring if available |                |
|                |                |                         | administrative drops are    | or justify why LTFU is |                |
|                |                |                         | common. Assuming zero loss  | negligible in your     |                |
|                |                |                         | to follow-up is             | specific datasets.     |                |
|                |                |                         | statistically \"brave\" and |                        |                |
|                |                |                         | likely incorrect.           |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.3-02     | Page 6,        | Equation (2): The       | This is the correct         | Add a brief note that  | Low            |
|                | Equation 2     | log-transform h = -(1 - | \"complementary log-log\"   | the weekly bin is      |                |
|                |                | MR)                     | link for                    | narrow enough that the |                |
|                |                |                         | discrete-to-continuous      | hazard is              |                |
|                |                |                         | hazard mapping. However,    | approximately constant |                |
|                |                |                         | for a reviewer, the use of  | within the interval.   |                |
|                |                |                         | \"piecewise-constant\" must |                        |                |
|                |                |                         | be defended for the chosen  |                        |                |
|                |                |                         | bin width (weeks).          |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.3-03     | Page 6, Line   | \"\...likelihood-based  | A biostatistician will ask  | Briefly explain the    | Medium         |
|                | 17 of this     | or partial-likelihood   | why you are avoiding the    | benefit of this        |                |
|                | section (para  | formulations are not    | efficiency of Maximum       | \"non-parametric\"     |                |
|                | after Eq. 2)   | used\...\"              | Likelihood Estimation       | start: it avoids       |                |
|                |                |                         | (MLE). It suggests the      | assuming a functional  |                |
|                |                |                         | method is purely \"curve    | form for \$h(t)\$      |                |
|                |                |                         | fitting\" rather than       | before the frailty     |                |
|                |                |                         | statistical inference.      | normalization.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.3-04     | Page 6, Last   | Nelson--Aalen           | If the results are          | Move the Nelson-Aalen  | Medium         |
|                | two lines      | sensitivity check.      | \"unchanged\" from the      | comparison to the      |                |
|                |                |                         | Nelson-Aalen estimator,     | \"Validation\" section |                |
|                |                |                         | then Equation (2) is        | to prove the           |                |
|                |                |                         | effectively just a          | estimator\'s           |                |
|                |                |                         | complication.               | robustness, or explain |                |
|                |                |                         |                             | why Eq (2) is          |                |
|                |                |                         |                             | preferred (e.g.,       |                |
|                |                |                         |                             | better behavior at     |                |
|                |                |                         |                             | high risk).            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.3-05     | Page 6, Line   | \"Bin width is chosen   | This is subjective. A       | Provide an objective   | Medium         |
|                | after Eq. 3    | based on diagnostic     | reviewer will worry that    | metric for             |                |
|                |                | stability.\"            | the author \"tuned\" the    | \"stability\" (e.g.,   |                |
|                |                |                         | bin width until the results | \"bins were chosen to  |                |
|                |                |                         | looked good.                | ensure a minimum of    |                |
|                |                |                         |                             | \$X\$ events per       |                |
|                |                |                         |                             | interval\").           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.4 Selection Model: Gamma Frailty and Depletion Normalization**                                                                |
+-----------------------------------------------------------------------------------------------------------------------------------+
| **2.4.1 Individual Hazards with Multiplicative Frailty**                                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.1-01   | Page 7, Eq (4) | z\_{i,d} sim {Gamma}(1, | The reviewers may ask:      | Add a sentence         | Medium         |
|                |                | theta_d)                | \"Why Gamma?\" If the true  | justifying Gamma via   |                |
|                |                |                         | frailty is bimodal or       | the \"Laplace          |                |
|                |                |                         | Log-normal, does the        | transform\" for        |                |
|                |                |                         | inversion break?            | tractability, but      |                |
|                |                |                         |                             | acknowledge it as a    |                |
|                |                |                         |                             | \"functional           |                |
|                |                |                         |                             | approximation.\"       |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.1-02   | Page 7, Eq (6) | The Log-transform       | This is the \"Vaupel        | Add citation to Vaupel | Medium         |
|                |                | identity H\_{obs} = 1 / | Identity.\" You must cite   | et al. (1979) or       |                |
|                |                | theta log(1+theta       | the original source (Vaupel | Hougaard (1984)        |                |
|                |                | tilde{H}\_0).           | et al. 1979) here, so       | immediately following  |                |
|                |                |                         | reviewers know you aren\'t  | Equation 6.            |                |
|                |                |                         | \"inventing\" the math but  |                        |                |
|                |                |                         | rather applying it.         |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.4.2 Gamma-frailty Identity and Inversion**                                                                                    |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.2-01   | Page 7, Line   | \"Depletion-neutralized | In standard Cox models,     | Consistently use the   | Medium         |
|                | after Eq. 6    | baseline cumulative     | \"baseline\" means          | term \"latent hazard\" |                |
|                |                | hazard\" terminology.   | \"covariates = 0.\" Here,   | or \"intrinsic         |                |
|                |                |                         | you mean \"population       | hazard\" alongside     |                |
|                |                |                         | before selection.\" This    | \"baseline\" to        |                |
|                |                |                         | shift in meaning can        | distinguish from the   |                |
|                |                |                         | confuse reviewers.          | Cox definition.        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.4.3 Baseline Shape Used for Frailty Identification**                                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.3-01   | Page 7, Eq (8) | Assumption:             | Major Identifiability Risk. | Explicitly label this  | High           |
|                |                | tilde{h}\_{0,d}(t) =    | This is the paper\'s \"weak | as the \"Identifying   |                |
|                |                | k_d (constant hazard).  | point.\" If the external    | Restriction.\" State   |                |
|                |                |                         | hazard is actually          | that the quiet window  |                |
|                |                |                         | declining (e.g., a pandemic | must be chosen where   |                |
|                |                |                         | wave subsiding), the model  | this assumption is     |                |
|                |                |                         | will \"force\" theta to be  | biologically           |                |
|                |                |                         | large to explain the        | plausible.             |                |
|                |                |                         | curvature.                  |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.3-02   | Page 7, First  | \"curvature is forced   | This is an honest           | Provide a sensitivity  | High           |
|                | line of last   | to be explained by      | description of your         | check in the SI        |                |
|                | para           | depletion\... rather    | \"Occam's Razor\" approach, | showing what happens   |                |
|                |                | than\... time-varying   | but a reviewer will call it | if the baseline hazard |                |
|                |                | baseline hazard.\"      | a \"strong structural       | is slightly linear     |                |
|                |                |                         | assumption.\"               | (sloped) rather than   |                |
|                |                |                         |                             | constant.              |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.3-03   | Page 7, Second | \"model naturally       | This is a great diagnostic  | Highlight this as a    | Low            |
|                | last line      | collapses toward hat    | property. If the data is    | \"safe failure mode\": |                |
|                |                | theta_d approx 0.\"     | linear, the model does      | if there is no         |                |
|                |                |                         | nothing.                    | selection, KCOR        |                |
|                |                |                         |                             | reduces to a standard  |                |
|                |                |                         |                             | cumulative contrast.   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.4.4 Quiet-Window Validity as the Key Dataset-Specific Requirement**                                                           |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.4-01   | Page 8, Line 1 | \"Prespecified quiet    | If the window is defined in | Clarify how you handle | High           |
|                |                | window (defined in      | calendar time but the       | the overlap of         |                |
|                |                | ISO-week space).\"      | analysis is in \"time since | calendar time (the     |                |
|                |                |                         | enrollment,\" different     | shock) and event time  |                |
|                |                |                         | individuals will be at      | (the depletion). A     |                |
|                |                |                         | different \"event-ages\"    | diagram showing this   |                |
|                |                |                         | during the window.          | intersection would be  |                |
|                |                |                         |                             | helpful.               |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.4-02   | Page 8, Line 3 | \"\...confound          | \"Geometry\" is again used  | Replace                | Low            |
|                |                | depletion-geometry      | as a catch-all. Reviewers   | \"depletion-geometry\" |                |
|                |                | estimation.\"           | prefer \"parameter          | with \"estimation of   |                |
|                |                |                         | estimation\" or \"frailty   | the frailty variance   |                |
|                |                |                         | variance estimation.\"      | (theta) and baseline   |                |
|                |                |                         |                             | hazard (k).\"          |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.4-03   | Page 8, Line 4 | Diagnostic (ii):        | This is essentially         | Explain that linearity | Medium         |
|                |                | \"post-normalization    | checking if the model fits  | outside the window     |                |
|                |                | linearity within the    | itself. It's a necessary    | (during other quiet    |                |
|                |                | window.\"               | but not sufficient          | weeks) is the stronger |                |
|                |                |                         | condition for validity.     | \"out-of-sample\"      |                |
|                |                |                         |                             | validation.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.4-04   | Page 8, Line 5 | Diagnostic (iii):       | This is a \"Leave-one-out\" | Mention that specific  | Medium         |
|                |                | \"stability\... to      | style check. Reviewers love | stability thresholds   |                |
|                |                | small boundary          | this, but they will want to | are defined (e.g., in  |                |
|                |                | perturbations.\"        | see the actual sensitivity  | the SI) to prevent     |                |
|                |                |                         | values (e.g., \"pm 10%      | arbitrary window       |                |
|                |                |                         | change in theta\").         | selection.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.4-05   | Page 8, Line 6 | \"diagnostics indicate  | Highly Commendable. This    | Maintain this stance.  | Low            |
|                |                | non-identifiability\... | level of transparency       | It is your strongest   |                |
|                |                | treated as not          | (refusing to report a       | defense against        |                |
|                |                | identified.\"           | result if the model         | \"over-fitting\"       |                |
|                |                |                         | doesn\'t fit) builds        | critiques.             |                |
|                |                |                         | significant trust with a    |                        |                |
|                |                |                         | reviewer.                   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.4-06   | Page 8, Line   | \"Quiet-window          | The term \"operational\"    | Change to              | Low            |
|                | 10             | selection protocol      | sounds like a manual.       | \"Operationalization   |                |
|                |                | (operational).\"        |                             | and Reproducibility\"  |                |
|                |                |                         |                             | to signal that anyone  |                |
|                |                |                         |                             | can replicate your     |                |
|                |                |                         |                             | window choice.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.4.4-07   | Page 8, Line   | Reference to SI §S2     | Since the quiet window is   | Summarize the *top     | High           |
|                | 11             | (Tables S1--S3).        | the \"Achilles\' heel\" of  | three* criteria (e.g., |                |
|                |                |                         | the paper, moving the       | R\^2 \> 0.98, minimal  |                |
|                |                |                         | criteria for it to the SI   | residual slope, etc.)  |                |
|                |                |                         | is risky.                   | in the main text.      |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.5 Estimation During Quiet Periods (Cumulative-Hazard Least Squares)**                                                         |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.5-01     | Page 8, Eq (9) | The functional form     | Major Identifiability Trap. | Explicitly state the   | High           |
|                |                | H\_{model} = 1 theta    | In a short window, a log    | minimum required       |                |
|                |                | log (1+ theta k t)      | curve can be very           | \"event density\" or   |                |
|                |                |                         | well-approximated by a      | \"window length\"      |                |
|                |                |                         | straight line. If the       | needed to break the    |                |
|                |                |                         | window isn\'t long enough   | collinearity between k |                |
|                |                |                         | to show distinct curvature, | and theta.             |                |
|                |                |                         | k and theta become highly   |                        |                |
|                |                |                         | collinear (multiple pairs   |                        |                |
|                |                |                         | of (k, theta will fit the   |                        |                |
|                |                |                         | data equally well).         |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.5-02     | Page 8, 4^th^  | \"Near-linear observed  | This is a \"safe\" failure  | Acknowledge that the   | Medium         |
|                | line after Eq  | cumulative hazards      | mode, but it assumes the    | theta ge 0 constraint  |                |
|                | (9)            | naturally drive\... hat | baseline is truly constant. | assumes all observed   |                |
|                |                | theta_d toward zero.\"  | If the hazard is slightly   | curvature is           |                |
|                |                |                         | *increasing* due to an      | \"downward\"           |                |
|                |                |                         | external factor, hat theta  | (concave). Explain     |                |
|                |                |                         | will be negative (which you | what happens if the    |                |
|                |                |                         | constrain to 0),            | data is convex.        |                |
|                |                |                         | potentially masking bias.   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.5-03     | Page 8, First  | \"Fitting is performed  | Statistical Efficiency      | Justify the use of LS  | High           |
|                | line after Eq  | in cumulative-hazard    | Concern. Least-squares on   | over Weighted Least    |                |
|                | (10)           | space rather than via   | cumulative sums (which are  | Squares or MLE by      |                |
|                |                | likelihood.\"           | autocorrelated) violates    | citing its             |                |
|                |                |                         | the independence assumption | \"robustness to        |                |
|                |                |                         | of standard OLS. This leads | instantaneous noise\"  |                |
|                |                |                         | to underestimated standard  | or \"structural shape  |                |
|                |                |                         | errors and biased point     | focus.\"               |                |
|                |                |                         | estimates in some settings. |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.5-04     | Page 8, 4^th^  | \"RMSE\... directly     | RMSE measures \"goodness of | Clarify that you also  | Medium         |
|                | line after Eq  | assess                  | fit,\" not                  | monitor parameter      |                |
|                | (10)           | identifiability.\"      | \"identifiability.\" A      | stability/variance     |                |
|                |                |                         | model can fit perfectly     | (e.g., via bootstrap   |                |
|                |                |                         | (low RMSE) but still have   | or the Jacobian) to    |                |
|                |                |                         | unidentifiable parameters   | ensure the parameters  |                |
|                |                |                         | if the Hessian is singular. | are uniquely           |                |
|                |                |                         |                             | determined.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.5-05     | Page 8, 3^rd^  | \"most commonly         | You are noting that theta   | Ensure you demonstrate | Medium         |
|                | Para of the    | observed in vaccinated  | approx. 0 for one group. If | (perhaps in SI) that   |                |
|                | section, First | cohorts\...\"           | theta is zero for one group | this difference in hat |                |
|                | line           |                         | but large for another, KCOR | theta is driven by the |                |
|                |                |                         | will apply a massive        | data\'s shape and not  |                |
|                |                |                         | correction to one but not   | a model artifact.      |                |
|                |                |                         | the other. This creates the |                        |                |
|                |                |                         | \"KCOR effect.\"            |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.6 Normalization (Depletion-Neutralized Cumulative Hazards)**                                                                  |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6-01     | Page 9, Eq     | Application of the      | Major Extrapolation Risk.   | Explicitly state that  | High           |
|                | (11)           | inversion to the full   | You estimated the selection | this assumes the       |                |
|                |                | trajectory.             | parameters in a \"quiet     | latent frailty         |                |
|                |                |                         | window,\" but you are       | distribution is a      |                |
|                |                |                         | applying them to the entire | \"fixed trait\" of the |                |
|                |                |                         | study. If the frailty       | cohort assigned at     |                |
|                |                |                         | distribution changes (e.g., | enrollment.            |                |
|                |                |                         | due to a new variant or     |                        |                |
|                |                |                         | different causes of death), |                        |                |
|                |                |                         | the inversion becomes       |                        |                |
|                |                |                         | invalid outside the window. |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6-02     | Page 9, Line 4 | Claim: \"\...hazard     | This is a strong geometric  | Clarify that only      | Medium         |
|                |                | curvature has been      | claim. A reviewer will      | curvature matching the |                |
|                |                | factored out.\"         | check if you\'ve also       | specific               |                |
|                |                |                         | factored out \"real\"       | Gamma-depletion        |                |
|                |                |                         | curvature (e.g., an actual  | profile is removed,    |                |
|                |                |                         | waning of vaccine           | leaving other shapes   |                |
|                |                |                         | effectiveness) by mistake.  | intact.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6-03     | Page 9, Line 5 | \"\...not equivalent to | This is a subtle point that | Add a brief sentence   | Low            |
|                |                | Cox partial-likelihood  | most readers will miss. You | explaining that KCOR   |                |
|                |                | baseline anchoring.\"   | are comparing               | compares the           |                |
|                |                |                         | \"population-level          | \"potential\" of the   |                |
|                |                |                         | intrinsic hazards,\" which  | groups, similar to an  |                |
|                |                |                         | is different from           | \"Intention-to-Treat\" |                |
|                |                |                         | \"conditional hazards\" in  | baseline.              |                |
|                |                |                         | Cox.                        |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6-04     | Page 9, Line 6 | \"In this space,        | This is the \"Strongest     | Soften the language:   | Medium         |
|                |                | cumulative hazards are  | Claim\" of the paper. It    | \"\...remaining        |                |
|                |                | directly                | rests entirely on the       | differences are more   |                |
|                |                | comparable\...\"        | assumption that             | likely to reflect real |                |
|                |                |                         | Gamma-frailty is the only   | differences in risk    |                |
|                |                |                         | reason for                  | rather than            |                |
|                |                |                         | non-proportionality.        | selection-induced      |                |
|                |                |                         |                             | artifacts.\"           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6-05     | Page 9, Line 8 | Reference to Equations  | This is helpful for the     | Consider a             | Low            |
|                |                | 2, 10, 11, and 1.       | reader, but as a reviewer,  | flow-diagram or a      |                |
|                |                |                         | I want to see a             | numbered list that     |                |
|                |                |                         | \"Walkthrough\" of the      | traces a single data   |                |
|                |                |                         | logic: observed-to-log,     | point through these    |                |
|                |                |                         | log-to-fit,                 | four specific          |                |
|                |                |                         | fit-to-inversion,           | equations.             |                |
|                |                |                         | inversion-to-ratio.         |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.6.1 Computational Considerations**                                                                                            |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6.1-01   | Page 9, Line 2 | Scaling linearly with   | Major Efficiency Advantage. | Lean into this. It     | Low            |
|                | on this        | time bins rather than   | This is a huge selling      | positions KCOR as a    |                |
|                | section        | individuals.            | point for registry data. It | \"Pre-processing       |                |
|                |                |                         | means the method is         | Utility\" that can run |                |
|                |                |                         | \"O(T)\" where T is the     | on a laptop even for   |                |
|                |                |                         | number of weeks, making it  | national-scale data.   |                |
|                |                |                         | essentially instantaneous   |                        |                |
|                |                |                         | compared to a Cox model     |                        |                |
|                |                |                         | which is \"O(N)\"           |                        |                |
|                |                |                         | (individuals).              |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6.1-02   | Page 9, Line 4 | \"Memory-bound rather   | This is a technical nuance  | Clarify that the       | Medium         |
|                | on this        | than CPU-bound.\"       | that might worry a          | memory bottleneck is   |                |
|                | section        |                         | systems-reviewer. Why is a  | only in the *initial*  |                |
|                |                |                         | discrete-time aggregation   | aggregation of raw     |                |
|                |                |                         | memory-heavy?               | records into bins;     |                |
|                |                |                         |                             | once binned, the KCOR  |                |
|                |                |                         |                             | math itself has a      |                |
|                |                |                         |                             | negligible memory      |                |
|                |                |                         |                             | footprint.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.6.2 Internal Diagnostics and 'Self-Check' Behavior**                                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6.2-01   | Page 9, Line 3 | \"\...should be         | Major Subjectivity Risk.    | Define a specific      | High           |
|                | of this        | approximately linear in | \"Approximately linear\" is | metric for linearity,  |                |
|                | section        | event time.\"           | not a statistical test. A   | such as an R-squared   |                |
|                |                |                         | reviewer will want to see a | threshold or a maximum |                |
|                |                |                         | formal test for linearity   | allowable deviation    |                |
|                |                |                         | (e.g., a test on the second | from the slope.        |                |
|                |                |                         | derivative or a lack-of-fit |                        |                |
|                |                |                         | test).                      |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6.2-02   | Page 9, First  | Residuals are not       | In survival analysis,       | Suggest using a        | Medium         |
|                | line after Eq  | \"systematically        | residuals are almost always | Durbin-Watson test or  |                |
|                | (12)           | time-structured.\"      | time-structured if the      | a similar              |                |
|                |                |                         | model is slightly off. You  | autocorrelation check  |                |
|                |                |                         | are essentially asking for  | to prove that          |                |
|                |                |                         | \"white noise\" residuals   | residuals are indeed   |                |
|                |                |                         | in a cumulative space.      | \"unstructured.\"      |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6.2-03   | Page 9, Item   | Stability to window     | Major Sensitivity Issue. If | State a \"Stability    | High           |
|                | 3. Parameter   | perturbations (e.g.,    | shifting the window by 4    | Criterion\" (e.g., the |                |
|                | stability to   | +/- 4 weeks).           | weeks changes the frailty   | frailty variance must  |                |
|                | window         |                         | variance by 50%, the entire | change by less than X% |                |
|                | perturbations  |                         | KCOR(t) curve will shift.   | when the window is     |                |
|                |                |                         | This is the \"fragility\"   | shifted).              |                |
|                |                |                         | point of the estimator.     |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6.2-04   | Page 9, Item 4 | Non-identifiability     | This is actually a          | Distinguish between    | Low            |
|                |                | when frailty variance   | \"success\" state for the   | \"Zero Depletion\"     |                |
|                |                | approaches zero.        | model (it means no          | (identifiable but      |                |
|                |                |                         | correction is needed).      | zero) and \"Numerical  |                |
|                |                |                         | Calling it                  | Instability\" (where   |                |
|                |                |                         | \"non-identifiability\"     | the optimizer doesn\'t |                |
|                |                |                         | might be confusing.         | converge).             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.6.2-05   | Page 10, Last  | \"KCOR should not be    | This is an excellent        | Explicitly list the    | Medium         |
|                | line of this   | interpreted\" if        | scientific boundary. It     | \"Failure to Launch\"  |                |
|                | section        | diagnostics fail.       | prevents the method from    | criteria in a summary  |                |
|                |                |                         | being used in \"un-quiet\"  | table so reviewers can |                |
|                |                |                         | periods.                    | see the \"Go/No-Go\"   |                |
|                |                |                         |                             | logic clearly.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.7 Stabilization (Early Weeks)**                                                                                               |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.7-01     | Page 10, Line  | \"excluding early       | Major Structural Risk. If   | Add a technical note   | High           |
|                | 3 of this      | weeks\... from          | you exclude the early       | explaining that the    |                |
|                | section        | quiet-window fitting.\" | weeks, you are estimating   | inversion math         |                |
|                |                |                         | frailty (theta) on a        | accounts for the       |                |
|                |                |                         | population that has already | \"starting point\" of  |                |
|                |                |                         | undergone its most intense  | the stabilization      |                |
|                |                |                         | period of depletion. This   | window to ensure theta |                |
|                |                |                         | could lead to an            | remains consistent.    |                |
|                |                |                         | underestimation of the true |                        |                |
|                |                |                         | frailty variance.           |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.7-02     | Page 10, Line  | \"exclude early         | Reviewers are naturally     | State that the         | Medium         |
|                | 4 of this      | enrollment instability  | suspicious of \"skip        | SKIP_WEEKS value is    |                |
|                | section        | rather than to tune     | windows.\" They may worry   | chosen based on prior  |                |
|                |                | estimates.\"            | that \"Week 1\" was skipped | knowledge of           |                |
|                |                |                         | because it didn\'t fit the  | administrative delays  |                |
|                |                |                         | model\'s curve.             | or clinical deferral   |                |
|                |                |                         |                             | periods, not by        |                |
|                |                |                         |                             | looking at the outcome |                |
|                |                |                         |                             | data.                  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.7-03     | Page 10, Eq    | Setting the effective   | Major Interpretation Issue. | Ensure all plots are   | High           |
|                | (13)           | hazard to zero for      | By setting the early hazard | clearly labeled as     |                |
|                |                | early weeks.            | to zero, the \"Cumulative   | \"Cumulative Hazard    |                |
|                |                |                         | Hazard\" at week 5 is       | starting at Week X\"   |                |
|                |                |                         | actually just the sum of    | to avoid confusing the |                |
|                |                |                         | weeks 4 and 5. This makes   | reader with \"total    |                |
|                |                |                         | the Y-axis \"Cumulative     | mortality.\"           |                |
|                |                |                         | Hazard since Week X,\" not  |                        |                |
|                |                |                         | \"since enrollment.\"       |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.7-04     | Page 10, Line  | \"evaluated via         | This is your best defense.  | Reference a specific   | Low            |
|                | 4 of this      | sensitivity analysis.\" | If you can show that        | figure in the Results  |                |
|                | section        |                         | skipping 2, 3, or 4 weeks   | or SI that shows this  |                |
|                |                |                         | yields the same             | \"Skip Window          |                |
|                |                |                         | \"neutralized\" result, the | Sensitivity.\"         |                |
|                |                |                         | reviewer will trust the     |                        |                |
|                |                |                         | choice.                     |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.8 KCOR Estimator**                                                                                                            |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.8-01     | Page 10, Eq    | Ratio of neutralized    | Major Stability Concern.    | Add a warning that     | High           |
|                | (14)           | cumulative hazards.     | Ratios are inherently       | KCOR values are most   |                |
|                |                |                         | unstable when the           | reliable after a       |                |
|                |                |                         | denominator is small (e.g., | minimum number of      |                |
|                |                |                         | in the early weeks). A      | events have            |                |
|                |                |                         | small shift in the          | accumulated in both    |                |
|                |                |                         | \"neutralized\" hazard for  | cohorts.               |                |
|                |                |                         | Cohort B can cause the KCOR |                        |                |
|                |                |                         | value to explode or         |                        |                |
|                |                |                         | collapse.                   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.9 Uncertainty Quantification**                                                                                                |
+-----------------------------------------------------------------------------------------------------------------------------------+
| **2.9.1 Stratified Bootstrap Procedure**                                                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.9-01     | Page 10, Item  | Resampling cohort-time  | Major Statistical Validity  | Justify why an         | High           |
|                | 1 of this      | counts (Aggregated      | Issue. You are resampling   | aggregated bootstrap   |                |
|                | section        | Bootstrap).             | counts within time bins.    | is used over a \"Block |                |
|                |                |                         | This assumes that the       | Bootstrap\" or         |                |
|                |                |                         | number of deaths in Week 5  | individual-level       |                |
|                |                |                         | is independent of Week 4.   | resampling, and        |                |
|                |                |                         | In reality, these counts    | acknowledge the        |                |
|                |                |                         | are linked by the risk set. | potential for          |                |
|                |                |                         | This \"Poisson-style\"      | underestimating        |                |
|                |                |                         | resampling may              | temporal correlation.  |                |
|                |                |                         | underestimate true          |                        |                |
|                |                |                         | variance.                   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.9-02     | Page 10, Item  | Re-estimating frailty   | Major Success. This is the  | Emphasize that your    | Low            |
|                | 2 of this      | parameters per          | correct way to propagate    | confidence intervals   |                |
|                | section        | bootstrap.              | \"Parameter Uncertainty.\"  | capture the \"fitting  |                |
|                |                |                         | By refitting the quiet      | error\" of the quiet   |                |
|                |                |                         | window for every skip, you  | window, not just the   |                |
|                |                |                         | are showing how sensitive   | randomness of the      |                |
|                |                |                         | the final ratio is to the   | deaths.                |                |
|                |                |                         | initial \"depletion\"       |                        |                |
|                |                |                         | guess.                      |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.9-03     | Page 10, Item  | Use of Percentile       | Percentile intervals (e.g., | Consider reporting     | Medium         |
|                | 5 of this      | Intervals.              | 2.5th to 97.5th) can be     | \"Bias-Corrected and   |                |
|                | section        |                         | biased if the bootstrap     | Accelerated\" (BCa)    |                |
|                |                |                         | distribution is skewed,     | intervals to account   |                |
|                |                |                         | which is common with        | for the skewness       |                |
|                |                |                         | ratios.                     | inherent in            |                |
|                |                |                         |                             | ratio-based            |                |
|                |                |                         |                             | estimators.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.9-04     | Page 11, 2^nd^ | \"Uncertainty intervals | A reviewer will ask: Does   | Clarify that the       | Medium         |
|                | last line of   | reflect\... model-fit   | this include the            | window boundaries are  |                |
|                | this section   | uncertainty.\"          | uncertainty of choosing the | fixed, but the         |                |
|                |                |                         | quiet window? If the window | parameters within      |                |
|                |                |                         | itself is fixed across all  | those boundaries are   |                |
|                |                |                         | bootstraps, you are         | allowed to vary.       |                |
|                |                |                         | under-representing the      |                        |                |
|                |                |                         | \"Researcher Choice\"       |                        |                |
|                |                |                         | error.                      |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.10 Algorithm Summary and Reproducibility Checklist**                                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.10-01    | Figure 1       | \"late-time curvature   | Consistency Error. Sections | Change \"late-time     | High           |
|                | Caption        | is used to estimate     | 1.2 and 2.5 emphasize using | curvature\" to         |                |
|                |                | frailty parameters.\"   | a \"quiet window.\" In many | \"quiet-window         |                |
|                |                |                         | datasets, the quiet window  | curvature\" to         |                |
|                |                |                         | is in the middle or early,  | maintain consistency   |                |
|                |                |                         | not \"late.\"               | with the rest of the   |                |
|                |                |                         |                             | manuscript.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.10-02    | Figure 1       | \"approximately         | Interpretation Issue. If    | Clarify that it is     | High           |
|                | (Panel B)      | linearized cumulative   | the normalization results   | linearized relative to |                |
|                |                | hazards.\"              | in a straight line, it      | the selection effect,  |                |
|                |                |                         | implies the hazard is now   | or that it is linear   |                |
|                |                |                         | constant. If the real       | within the quiet       |                |
|                |                |                         | hazard is actually rising   | window only.           |                |
|                |                |                         | (e.g., winter mortality),   |                        |                |
|                |                |                         | it shouldn\'t be linear.    |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.10-03    | Figure 1       | Schematic               | Schematics often show       | Ensure the schematic   | Medium         |
|                | (Schematic)    | \"perfection.\"         | perfectly clean lines. If   | includes at least a    |                |
|                |                |                         | your actual results in      | small amount of        |                |
|                |                |                         | Section 3 show \"wobbly\"   | \"noise\" or           |                |
|                |                |                         | lines, the reviewer will    | \"shocks\" to show how |                |
|                |                |                         | feel the schematic was      | KCOR handles non-ideal |                |
|                |                |                         | over-simplified.            | data.                  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.10-04    | Figure 1       | \"KCOR\... is near-flat | This is your \"Main Selling | Change to \"is         | Low            |
|                | Caption        | under the null.\"       | Point.\" You must be        | expected to be         |                |
|                |                |                         | careful---even under the    | horizontal under the   |                |
|                |                |                         | null, KCOR will have        | null, subject to       |                |
|                |                |                         | \"wiggles\" due to sampling | sampling               |                |
|                |                |                         | error.                      | stochasticity.\"       |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.11 Relationship to Cox Proportional Hazards**                                                                                 |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11-01    | Page 11, Last  | \"Cox targets a         | Major Logical Hurdle. Most  | Clarify that the       | High           |
|                | Line           | different quantity\...  | reviewers believe the       | scientific question    |                |
|                |                | than KCOR's cumulative  | \"Instantaneous Hazard      | is: \"What is the      |                |
|                |                | hazard estimand.\"      | Ratio\" is the gold         | inherent risk          |                |
|                |                |                         | standard. If you claim it   | difference between     |                |
|                |                |                         | doesn\'t align with the     | these groups, stripped |                |
|                |                |                         | \"scientific question,\"    | of selection           |                |
|                |                |                         | you must define that        | artifacts?\"           |                |
|                |                |                         | question explicitly.        |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11-02    | Page 12, Line  | \"\...Cox results are   | This is a very bold move.   | Ensure the Results     | Medium         |
|                | 2              | presented here as a     | You are effectively         | section clearly shows  |                |
|                |                | diagnostic              | demoting the world\'s most  | exactly how Cox fails  |                |
|                |                | demonstration\... not   | popular survival model to a | (e.g., by showing a    |                |
|                |                | as a competing          | \"bias detector.\"          | crossing or declining  |                |
|                |                | estimator.\"            |                             | hazard ratio under a   |                |
|                |                |                         |                             | null effect) to        |                |
|                |                |                         |                             | justify this stance.   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11-03    | Page 12, Line  | \"Even when Cox models  | Major Technical             | Explain that KCOR is   | High           |
|                | 5              | are extended with       | Distinction. A Frailty Cox  | more robust because it |                |
|                |                | shared frailty\... they | model tries to estimate a   | only uses the frailty  |                |
|                |                | continue to estimate    | single hazard ratio that is | model to normalize the |                |
|                |                | instantaneous hazard    | \"true\" for every          | data, not to dictate   |                |
|                |                | ratios.\"               | individual. KCOR doesn\'t   | the final effect       |                |
|                |                |                         | care about the individual   | estimate.              |                |
|                |                |                         | level; it aims to \"fix\"   |                        |                |
|                |                |                         | the cohort\'s aggregate     |                        |                |
|                |                |                         | path.                       |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11-04    | Page 12, Line  | \"KCOR uses a           | This is the                 | Highlight this as a    | Low            |
|                | 9              | parametric working      | \"Semi-Parametric\"         | \"Best of Both         |                |
|                |                | model only to           | defense. You use the math   | Worlds\" approach:     |                |
|                |                | normalize\... then      | of frailty to clean the     | parametric power for   |                |
|                |                | computes a cumulative   | data, but the final         | cleaning,              |                |
|                |                | contrast.\"             | comparison is               | non-parametric safety  |                |
|                |                |                         | non-parametric.             | for the final result.  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11-05    | Page 12,       | Lack of mention of      | Statisticians distinguish   | Use the terminology    | Medium         |
|                | General        | \"Marginal vs.          | between \"Marginal\"        | \"neutralized marginal |                |
|                |                | Conditional.\"          | (population-level) and      | estimand\" to help     |                |
|                |                |                         | \"Conditional\"             | statisticians place    |                |
|                |                |                         | (individual-level) effects. | KCOR in the existing   |                |
|                |                |                         | KCOR is essentially a       | literature.            |                |
|                |                |                         | neutralized marginal        |                        |                |
|                |                |                         | effect.                     |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.11.1 Demonstration: Cox Bias Under Frailty Heterogeneity with No Treatment Effect**                                           |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11.1-01  | Page 13,       | Cox hazard ratio drops  | This demonstrates a         | Emphasize that this    | Low            |
|                | Figure 2       | to approximately 0.55   | catastrophic structural     | bias is an inherent    |                |
|                |                | as frailty variance     | failure where a standard    | property of the Cox    |                |
|                |                | (theta) increases.      | model reports a significant | estimand under         |                |
|                |                |                         | treatment effect when the   | depletion, not a       |                |
|                |                |                         | true effect is zero.        | matter of sample size. |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11.1-02  | Page 14,       | KCOR shows a slight     | A reviewer may ask if the   | Note that while KCOR   | Medium         |
|                | Figure 3       | upward trend (error of  | Gamma-frailty model begins  | is not perfectly flat  |                |
|                |                | 2.5 percent) at the     | to \"drift\" or             | at extreme values, its |                |
|                |                | extreme theta of 20.    | over-correct when           | error is negligible    |                |
|                |                |                         | population heterogeneity is | compared to the 45     |                |
|                |                |                         | extremely high.             | percent error seen in  |                |
|                |                |                         |                             | Cox.                   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11.1-03  | Page 12,       | Use of Nelson-Aalen for | This is a vital             | Explicitly highlight   | Low            |
|                | Section 2.11.1 | KCOR instead of         | \"Information Constraint\"  | that KCOR maintains    |                |
|                |                | individual simulator    | test. It proves KCOR works  | accuracy even when it  |                |
|                |                | data.                   | with the limited data       | lacks access to the    |                |
|                |                |                         | available in real-world     | individual frailty     |                |
|                |                |                         | registries.                 | values used to         |                |
|                |                |                         |                             | generate the data.     |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11.1-04  | Page 12,       | Experiment includes     | KCOR\'s identification      | State clearly that     | High           |
|                | Section 2.11.1 | both constant and       | relies on a constant hazard | KCOR is robust to      |                |
|                |                | Gompertz baseline       | assumption during the quiet | non-constant baselines |                |
|                |                | hazards.                | window (§2.4.3). A Gompertz | as long as the slope   |                |
|                |                |                         | slope could potentially     | is gradual relative to |                |
|                |                |                         | interfere with this.        | the depletion rate.    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.11.1-05  | Page 13,       | Bootstrap intervals for | With \"millions of          | Ensure the text        | Medium         |
|                | Figure 3       | KCOR are described as   | individuals\" in the        | distinguishes between  |                |
|                |                | \"narrow\".             | simulation, stochastic      | \"Sampling             |                |
|                |                |                         | noise is very low. A        | Uncertainty\" and the  |                |
|                |                |                         | reviewer will want to know  | \"Parameter            |                |
|                |                |                         | if the intervals reflect    | Uncertainty\" captured |                |
|                |                |                         | \"Model Uncertainty\"       | by refitting the model |                |
|                |                |                         | correctly.                  | in each bootstrap.     |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.12 Worked Example (Descriptive)**                                                                                             |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.12-01    | Page 15,       | The worked example is   | Reviewers may worry that a  | Explicitly state if    | Low            |
|                | Section 2.12   | \"descriptive\" and     | simplified descriptive      | the example handles    |                |
|                |                | intended solely to      | example ignores real-world  | discrete-time ties or  |                |
|                |                | demonstrate mechanics.  | data issues such as         | sparse bins to prove   |                |
|                |                |                         | \"tied\" events or          | the mechanics are      |                |
|                |                |                         | reporting lags that         | robust for \"messy\"   |                |
|                |                |                         | complicate the hazard       | registry data.         |                |
|                |                |                         | estimation step.            |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.12-02    | Page 15,       | Mention of \"diagnostic | These plots are the only    | Ensure the worked      | Medium         |
|                | Section 2.12   | plots\" for linearity   | way to verify the           | example figures are    |                |
|                |                | and stability.          | \"self-check\" behavior     | explicitly             |                |
|                |                |                         | described in earlier        | cross-referenced here  |                |
|                |                |                         | sections.                   | so the reviewer can    |                |
|                |                |                         |                             | immediately see the    |                |
|                |                |                         |                             | \"before and after\"   |                |
|                |                |                         |                             | of the gamma           |                |
|                |                |                         |                             | inversion.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **2.13 Reproducibility and Computational Implementation**                                                                         |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.13-01    | Page 15,       | \"All figures, tables,  | While computational         | Provide a small,       | High           |
|                | Section 2.13   | and simulations can be  | reproducibility is a major  | synthetic \"dummy      |                |
|                |                | reproduced from the     | strength, registry data is  | dataset\" in the       |                |
|                |                | accompanying code       | often restricted. Reviewers | repository that allows |                |
|                |                | repository\".           | cannot \"make paper-full\"  | reviewers to run the   |                |
|                |                |                         | without the raw input data. | full Makefile pipeline |                |
|                |                |                         |                             | to verify the logic    |                |
|                |                |                         |                             | independently.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| MET-2.13-02    | Page 15,       | Use of a \"root         | Relying on a specific build | Briefly list the       | Low            |
|                | Section 2.13   | Makefile paper target\" | environment (Makefile) can  | primary language       |                |
|                |                | for the build process.  | be a hurdle for reviewers   | requirements (e.g.,    |                |
|                |                |                         | who use different software  | Python, R) in the main |                |
|                |                |                         | stacks (e.g., pure R or     | text to signal the     |                |
|                |                |                         | Python environments).       | technical              |                |
|                |                |                         |                             | accessibility of the   |                |
|                |                |                         |                             | code.                  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **3. RESULTS**                                                                                                                    |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1-01     | Page 15,       | Introduction of KCOR    | This is a significant       | Justify why this       | High           |
|                | Section 3.1    | relative to an early    | change in the estimand      | \"relative\" version   |                |
|                |                | post-enrollment         | (switching from an absolute | is necessary if the    |                |
|                |                | reference.              | ratio to a relative change  | normalization is       |                |
|                |                |                         | over time). It implies that | supposed to be         |                |
|                |                |                         | the early \"enrollment      | \"exact\" under the    |                |
|                |                |                         | noise\" is too high to      | null.                  |                |
|                |                |                         | normalize perfectly.        |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **3.1 Negative Controls (Selection-Only Null)**                                                                                   |
|                                                                                                                                   |
| **3.1.1 Fully Synthetic Negative Control (In-Model Gamma-Frailty Null)**                                                          |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1.1-01   | Page 15,       | Data-generating process | This is a \"sanity check,\" | Explicitly label this  | Low            |
|                | Section 3.1.1  | \"exactly matches\" the | not a robustness test. It   | as a \"Mathematical    |                |
|                |                | working model.          | proves the code is correct, | Identity               |                |
|                |                |                         | but it does not prove the   | Verification\" to      |                |
|                |                |                         | model works when reality is | manage expectations    |                |
|                |                |                         | more complex than a Gamma   | before moving to       |                |
|                |                |                         | distribution.               | out-of-model tests.    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1.1-02   | Page 15,       | KCOR remains            | In a professional review,   | Provide the mean and   | Medium         |
|                | Section 3.1.1  | \"approximately         | \"approximately\" is a      | standard deviation of  |                |
|                |                | constant at 1\".        | vague term. A skeptic will  | the KCOR asymptote     |                |
|                | 2^nd^ para of  |                         | want to know the exact      | across the simulation  |                |
|                | this section   |                         | numerical deviation (e.g.,  | replicates to quantify |                |
|                |                |                         | within 0.01 percent) to     | the \"approximation\". |                |
|                |                |                         | ensure there is no hidden   |                        |                |
|                |                |                         | bias in the estimator.      |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1.1-03   | Page 15,       | Full design details and | Because this is the         | Include at least one   | Medium         |
|                | Section 3.1.1  | figures are moved to    | foundational \"Null Case,\" | representative plot    |                |
|                |                | the Supplementary       | a reviewer may feel that    | from Figure S3 in the  |                |
|                | 2^nd^ para of  | Information.            | hiding the \"theta\" grid   | main text to visually  |                |
|                | this section   |                         | and the baseline hazard     | anchor the \"linearity |                |
|                |                |                         | shapes in the SI makes it   | after normalization\"  |                |
|                |                |                         | harder to judge the         | claim.                 |                |
|                |                |                         | experiment\'s difficulty.   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **3.1.2 Empirical Negative Control Using National Registry Data (Czech Republic)**                                                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1.2-01   | Page 15, First | Use of age-shift        | Age is a biological driver  | Briefly explain that   | Medium         |
|                | para of this   | (pseudo-cohorts) to     | of baseline hazard, not     | while age is           |                |
|                | section        | simulate selection      | just a \"selection          | biological, the        |                |
|                |                | effects.                | artifact.\" A reviewer may  | resulting hazard       |                |
|                |                |                         | argue that modeling         | curvature in the       |                |
|                |                |                         | age-related mortality as    | population\'s survival |                |
|                |                |                         | \"depletion\" is a          | curve can still be     |                |
|                |                |                         | mathematical convenience    | effectively            |                |
|                |                |                         | rather than a reflection of | neutralized by the     |                |
|                |                |                         | underlying biology.         | same Gamma-frailty     |                |
|                |                |                         |                             | geometric framework.   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1.2-02   | Page 16,       | Use of anchored KCOR    | Major Transparency Risk. By | Include a small        | High           |
|                | Figure 4       | starting at 4 weeks.    | anchoring at week 4, you    | \"inset\" or           |                |
|                |                |                         | are essentially \"zeroing   | supplementary figure   |                |
|                |                |                         | out\" the initial           | showing the unanchored |                |
|                |                |                         | cumulative hazard           | KCOR to demonstrate    |                |
|                |                |                         | difference. A skeptic will  | exactly how large the  |                |
|                |                |                         | worry that the model is     | \"pre-existing         |                |
|                |                |                         | hiding a large \"jump\" or  | cumulative             |                |
|                |                |                         | failure in the first 28     | differences\" were     |                |
|                |                |                         | days.                       | before normalization.  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1.2-03   | Page 16, Last  | Claim that asymmetry    | This is a strong defense    | Explicitly reference   | Low            |
|                | para before    | arises \"entirely from  | against \"model bias\"      | the \"Blind            |                |
|                | Figure 4       | differences in hazard   | claims. If the model        | Estimation\" protocol  |                |
|                |                | curvature\".            | doesn\'t know who is in     | here to remind the     |                |
|                |                |                         | which group, the correction | reviewer that the      |                |
|                |                |                         | is objective.               | model fits parameters  |                |
|                |                |                         |                             | cohort-by-cohort       |                |
|                |                |                         |                             | without cross-talk.    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.1.2-04   | Page 16,       | 95 percent bootstrap    | Large national datasets     | Mention that the       | Medium         |
|                | Figure 4       | intervals appear very   | have high statistical       | intervals specifically |                |
|                |                | narrow.                 | power, but narrow intervals | include the            |                |
|                |                |                         | can sometimes signal that   | re-estimation of the   |                |
|                |                |                         | the bootstrap is not        | frailty parameters for |                |
|                |                |                         | capturing the full          | every bootstrap        |                |
|                |                |                         | uncertainty of the \"Quiet  | iteration to prove     |                |
|                |                |                         | Window\" choice.            | they aren\'t \"too     |                |
|                |                |                         |                             | thin\".                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **3.2 Positive Controls (Injected Effects)**                                                                                      |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.2-01     | Page 16, First | Positive controls are   | Major Strategic Risk. A     | Move at least one      | High           |
|                | Line of this   | entirely displaced to   | reviewer may conclude that  | representative         |                |
|                | section        | Supplementary Section   | the main manuscript lacks   | \"Injected Harm\" plot |                |
|                |                | S3.                     | the evidence required to    | from Figure S1 into    |                |
|                |                |                         | prove the tool can detect   | the main body to       |                |
|                |                |                         | real harm or benefit. If    | demonstrate the        |                |
|                |                |                         | KCOR is \"too good\" at     | estimator\'s dynamic   |                |
|                |                |                         | returning 1.0 in the        | range.                 |                |
|                |                |                         | negative controls, the      |                        |                |
|                |                |                         | reader needs to see it      |                        |                |
|                |                |                         | \"moving\" in the main      |                        |                |
|                |                |                         | text.                       |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.2-02     | Page 16, Last  | Use of \"multiplicative | KCOR operates on a          | Explicitly state the   | Medium         |
|                | Line of this   | hazard shifts\" as the  | neutralized cumulative      | recovery accuracy      |                |
|                | page           | injected effect.        | scale. A reviewer will want | (e.g., \"recovered     |                |
|                |                |                         | to verify that a hazard     | within X percent of    |                |
|                |                |                         | shift injected at the       | the true injected      |                |
|                |                |                         | individual level is         | hazard ratio\") to     |                |
|                |                |                         | recovered linearly by the   | provide a quantitative |                |
|                |                |                         | KCOR ratio after            | performance metric.    |                |
|                |                |                         | normalization.              |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.2-03     | Page 16, End   | Mentions of             | In registry data,           | Quantify how much      | Medium         |
|                | of first para  | \"discretization and    | discretization (weekly      | \"discretization       |                |
|                | of this        | sampling noise\" as     | bins) can mask the exact    | noise\" contributes to |                |
|                | section        | reasons for deviation.  | timing of an effect. A      | the error compared to  |                |
|                |                |                         | reviewer might worry that   | the \"sampling noise\" |                |
|                |                |                         | KCOR \"smears\" the onset   | from the bootstrap.    |                |
|                |                |                         | of a new risk across        |                        |                |
|                |                |                         | multiple weeks.             |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.2-04     | Page 16, Last  | Claim that KCOR         | \"Reliably\" is a           | Define the lower bound | Low            |
|                | Line           | \"reliably detects both | subjective term. Reviewers  | of an effect that KCOR |                |
|                |                | harm and benefit\".     | prefer \"sensitivity\" or   | can distinguish from   |                |
|                |                |                         | \"minimum detectable effect | the null given the     |                |
|                |                |                         | size\".                     | typical noise levels   |                |
|                |                |                         |                             | in the Czech or        |                |
|                |                |                         |                             | similar datasets.      |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **3.3 Stress Tests (Frailty Misspecification)**                                                                                   |
|                                                                                                                                   |
| **3.3.1 Frailty Misspecification Robustness**                                                                                     |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.3.1-01   | Page 17, 3^rd^ | KCOR \"degrades         | Major Statistical Concern.  | Define the \"Direction | High           |
|                | para first     | gracefully\" by         | If misspecification leads   | of Bias\" for each     |                |
|                | line of this   | attenuating toward      | to attenuation, KCOR might  | misspecified           |                |
|                | section        | unity.                  | be \"under-correcting.\" A  | distribution. Does it  |                |
|                |                |                         | reviewer will want to know  | always fail toward the |                |
|                |                |                         | if this leads to a false    | null (conservative) or |                |
|                |                |                         | sense of security or        | can it overshoot       |                |
|                |                |                         | \"spurious nulls\".         | (anti-conservative)?   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.3.1-02   | Page 17, 3^rd^ | \"Non-identifiability   | This is a strong scientific | Provide a table        | Medium         |
|                | para of this   | rate\" reported as a    | defense. It shows that when | showing the rejection  |                |
|                | section        | performance metric.     | the model is wrong, it      | rate for each          |                |
|                |                |                         | \"knows\" it is wrong and   | distribution. If       |                |
|                |                |                         | refuses to provide an       | bimodal distributions  |                |
|                |                |                         | answer.                     | are rejected 90        |                |
|                |                |                         |                             | percent of the time,   |                |
|                |                |                         |                             | it proves the          |                |
|                |                |                         |                             | \"self-check\" is      |                |
|                |                |                         |                             | working.               |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.3.1-03   | Page 17, 3^rd^ | Comparison with         | These distributions are     | Use the term           | Low            |
|                | and 4^th^      | \"Two-point mixture\"   | significantly different     | \"Geometric            |                |
|                | items of the   | and \"Bimodal\"         | from the unimodal Gamma     | Robustness\" to        |                |
|                | list           | frailty.                | shape. Success here         | explain why a          |                |
|                |                |                         | suggests the \"Depletion    | Gamma-working-model    |                |
|                |                |                         | Geometry\" is more about    | can successfully       |                |
|                |                |                         | concavity than the specific | approximate other      |                |
|                |                |                         | distribution.               | concave depletion      |                |
|                |                |                         |                             | patterns.              |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| RES-3.3.1-04   | Page 17, Line  | Mention of \"residual   | This connects back to the   | Include a              | Medium         |
|                | 5 of the 3^rd^ | autocorrelation\" as a  | diagnostic methods. If the  | \"Representative       |                |
|                | Para of this   | failure signal.         | residuals are               | Failure\" plot in the  |                |
|                | section        |                         | autocorrelated, the model   | SI showing what a      |                |
|                |                |                         | has missed the \"shape\" of | failed diagnostic      |                |
|                |                |                         | the depletion.              | looks like for a       |                |
|                |                |                         |                             | bimodal distribution.  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **4. DISCUSSION**                                                                                                                 |
|                                                                                                                                   |
| **4.1 Limits of Attribution and Non-Identifiability**                                                                             |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.1-01     | Page 17, Line  | Admission that          | Major Inferential Risk. If  | Explicitly discuss how | High           |
|                | 2 of this      | curvature may arise     | seasonality or behavior     | the \"Quiet Window\"   |                |
|                | section        | from \"multiple         | change creates a curve      | strategy is            |                |
|                |                | sources\" (e.g.,        | similar to depletion, KCOR  | specifically designed  |                |
|                |                | behavior change,        | will \"neutralize\" it.     | to avoid these         |                |
|                |                | seasonality).           | This could accidentally     | confounders by         |                |
|                |                |                         | erase real signals or       | selecting periods      |                |
|                |                |                         | create false ones.          | where they are stable. |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.1-02     | Page 29, Last  | KCOR is \"not           | This is a safe statistical  | Reiterate that KCOR is | Medium         |
|                | Line of the    | attribution of that     | stance, but a public health | a \"Signal-to-Noise\"  |                |
|                | section        | curvature to a specific | reviewer might find it      | tool; it cleans the    |                |
|                |                | mechanism\".            | frustrating. They want to   | signal so that         |                |
|                |                |                         | know why the hazard is      | substantive            |                |
|                |                |                         | changing.                   | attribution can happen |                |
|                |                |                         |                             | more reliably          |                |
|                |                |                         |                             | afterward.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.1-03     | Page 29, Line  | Reliance on \"negative  | If the negative controls    | Cross-reference the    | Medium         |
|                | 4-5 of the     | controls\" to evaluate  | fail, the entire            | \"Failure Signaling\"  |                |
|                | section        | model adequacy.         | attribution-neutral stance  | diagnostics here to    |                |
|                |                |                         | collapses. The reviewer     | show that the method   |                |
|                |                |                         | needs to know the \"failure | prefers to return      |                |
|                |                |                         | protocol\".                 | \"Not Identified\"     |                |
|                |                |                         |                             | rather than            |                |
|                |                |                         |                             | \"Attributed to        |                |
|                |                |                         |                             | Depletion\"            |                |
|                |                |                         |                             | incorrectly.           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.1-04     | Page 29, Line  | \"Parsimonious working  | This is an excellent        | Keep this phrasing; it | Low            |
|                | 4 of the       | model\" vs.             | philosophical defense. It   | is the strongest       |                |
|                | section        | \"Substantive truth\".  | aligns with George Box's    | defense against        |                |
|                |                |                         | \"all models are wrong, but | biologists who might   |                |
|                |                |                         | some are useful\".          | challenge the          |                |
|                |                |                         |                             | Gamma-distribution     |                |
|                |                |                         |                             | assumption.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **4.2 What KCOR Estimates**                                                                                                       |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.2-01     | Page 18, First | KCOR is an              | Major Framing Risk. If KCOR | Explicitly state that  | Medium         |
|                | Line of Para   | \"intermediate layer\"  | is not a \"fully identified | while it doesn\'t      |                |
|                | 5,             | between descriptive     | effect estimator,\" critics | claim causal           |                |
|                |                | summaries and effect    | may argue it is just a      | attribution, it        |                |
|                |                | estimators.             | \"better visualization      | provides a             |                |
|                |                |                         | tool\" rather than a        | \"pre-requisite        |                |
|                |                |                         | rigorous statistical        | normalization\"        |                |
|                |                |                         | method.                     | without which causal   |                |
|                |                |                         |                             | inference is           |                |
|                |                |                         |                             | impossible.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.2-02     | Page 18, 2^nd^ | Admission that KCOR     | This is a crucial ethical   | Reiterate that KCOR    | Low            |
|                | last para of   | \"does not justify      | boundary. However, it may   | provides the \"clean   |                |
|                | the section    | claims about net lives  | disappoint public health    | comparison scale\"     |                |
|                |                | saved\".                | users who want to know the  | upon which other       |                |
|                |                |                         | \"bottom line\" of vaccine  | scientists can then    |                |
|                |                |                         | impact.                     | build survival-based   |                |
|                |                |                         |                             | summaries.             |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.2-03     | Page 18, 3^rd^ | Claim that frailty      | A reviewer will check if    | Clarify that the       | High           |
|                | last para of   | correction is           | this asymmetry is an        | asymmetry is detected  |                |
|                | the section    | \"asymmetric\" between  | assumption or a result. If  | from the data          |                |
|                |                | vaccinated and          | you assume vaccinated       | curvature, not imposed |                |
|                |                | unvaccinated.           | cohorts have lower frailty, | as a prior belief      |                |
|                |                |                         | the model is biased.        | about vaccination.     |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.2-04     | Page 27, Table | Comparison with         | Splines are very popular.   | Explain that splines   | Medium         |
|                | 3              | \"Flexible parametric   | The table says they         | treat the \"depletion  |                |
|                |                | survival (splines)\".   | \"absorb depletion          | signal\" as a          |                |
|                |                |                         | curvature.\" A reviewer     | \"biological change in |                |
|                |                |                         | will ask: \"Why is          | risk,\" leading to a   |                |
|                |                |                         | absorbing it a failure if   | misunderstanding of    |                |
|                |                |                         | the fit is good?\".         | the actual hazard.     |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.2-05     | Page 18, Last  | KCOR operates in        | This is a technical nuance. | Keep this explanation; | Low            |
|                | para of the    | cumulative-hazard space | Curvature in survival space | it justifies the move  |                |
|                | section        | because curvature is    | is                          | away from the          |                |
|                |                | \"additive\" there.     | multiplicative/exponential, | traditional \"survival |                |
|                |                |                         | making it harder to fit     | curve\" view.          |                |
|                |                |                         | with simple models.         |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **4.3 Relationship to Negative Control Methods**                                                                                  |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.3-01     | Page 18, Line  | KCOR \"normalizes       | Major Distinguishing Claim. | Explicitly contrast    | Medium         |
|                | 2 of this      | away\" confounding      | Standard negative controls  | KCOR with \"Negative   |                |
|                | section        | rather than just        | (like testing a vaccine     | Control Outcomes.\"    |                |
|                |                | detecting it.           | against a non-related cause | While a control        |                |
|                |                |                         | of death) only tell you     | outcome identifies     |                |
|                |                |                         | bias exists; they don\'t    | bias, KCOR\'s \"Quiet  |                |
|                |                |                         | fix it. KCOR claims to do   | Window\" uses the      |                |
|                |                |                         | both.                       | outcome of interest to |                |
|                |                |                         |                             | self-correct the bias. |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.3-02     | Page 19, 2^nd^ | \"Unvaccinated hazards  | This is the                 | Ensure the text        | High           |
|                | para of the    | are suppressed by       | \"Counter-Intuitive\" core  | clearly states that    |                |
|                | section        | ongoing frailty         | of the paper. Most people   | this suppression makes |                |
|                |                | depletion\".            | think depletion makes a     | the unvaccinated group |                |
|                |                |                         | group look *riskier*; you   | appear to have a       |                |
|                |                |                         | are explaining why it makes | \"decreasing hazard,\" |                |
|                |                |                         | them look *safer* (as the   | which is the           |                |
|                |                |                         | frailest die off).          | \"selection trap\".    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.3-03     | Page 19, Last  | Claim that unadjusted   | Major Policy Implication.   | Strengthen this by     | High           |
|                | line of the    | comparisons             | If true, this implies that  | cross-referencing the  |                |
|                | section        | \"systematically        | many reported Vaccine       | \"Synthetic Null\"     |                |
|                |                | understate unvaccinated | Effectiveness (VE)          | (Figure 2), which      |                |
|                |                | baseline risk\".        | estimates are structurally  | provides the           |                |
|                |                |                         | inflated.                   | mathematical proof of  |                |
|                |                |                         |                             | this inflation.        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.3-04     | Page 19, 2^nd^ | Mention of \"periods    | This refers to the          | Link this back to the  | Medium         |
|                | para Line 2 of | lacking a plausible     | \"Healthy Vaccinee Effect\" | \"Stabilization Skip   |                |
|                | the section    | mechanism\".            | seen immediately after      | Window\" (§2.7) to     |                |
|                |                |                         | vaccination. Reviewers will | show how KCOR handles  |                |
|                |                |                         | want to know if KCOR        | these distinct phases  |                |
|                |                |                         | handles the \"Early Week\"  | of bias.               |                |
|                |                |                         | noise differently than      |                        |                |
|                |                |                         | \"Long-term Depletion\".    |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **4.4 Practical Guidelines for Implementation**                                                                                   |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.4-01     | Page 19, Item  | \"Quiet-window          | Major Subjectivity Risk. If | Recommend that         | High           |
|                | 1              | definition and          | the window is chosen after  | researchers            |                |
|                |                | justification\".        | seeing the results, the     | pre-specify the quiet  |                |
|                |                |                         | analysis is compromised.    | window based on        |                |
|                |                |                         | Reviewers will demand that  | independent            |                |
|                |                |                         | the window be justified by  | epidemiological        |                |
|                |                |                         | external data (e.g.,        | surveillance data.     |                |
|                |                |                         | low-infection periods).     |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.4-02     | Page 19, Item  | \"Skip/stabilization    | The \"Skip Window\" (§2.7)  | Require a \"Stability  | Medium         |
|                | 5              | rule and robustness to  | is often the most           | Plot\" in the          |                |
|                |                | nearby values\".        | controversial part of       | reporting standard     |                |
|                |                |                         | vaccine studies. If results | that shows KCOR        |                |
|                |                |                         | only work with a 4-week     | estimates across a     |                |
|                |                |                         | skip but fail with a 3-week | range of skip-window   |                |
|                |                |                         | skip, the method is         | values.                |                |
|                |                |                         | fragile.                    |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.4-03     | Page 19, Item  | Baseline-shape choice   | While a constant baseline   | Suggest that if a      | Medium         |
|                | 4              | (default constant       | is parsimonious, it may be  | non-constant baseline  |                |
|                |                | baseline).              | a poor fit for diseases     | is used, the           |                |
|                |                |                         | with strong seasonality.    | researcher must        |                |
|                |                |                         | The \"justification\" for   | provide a sensitivity  |                |
|                |                |                         | this choice must be         | test comparing it to   |                |
|                |                |                         | rigorous.                   | the constant-baseline  |                |
|                |                |                         |                             | null.                  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| DIS-4.4-04     | Page 19, Last  | KCOR as a \"complete    | This is a strong defensive  | Explicitly state that  | Low            |
|                | para of the    | pipeline\" rather than  | stance. It prevents people  | KCOR should be applied |                |
|                | section        | a \"standalone          | from applying KCOR to data  | to raw aggregated      |                |
|                |                | adjustment\".           | that has already been       | counts from a          |                |
|                |                |                         | pre-filtered or adjusted by | \"frozen\" cohort to   |                |
|                |                |                         | other methods, which would  | maintain the integrity |                |
|                |                |                         | lead to \"double-counting\" | of the selection       |                |
|                |                |                         | bias.                       | geometry.              |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **5. LIMITATIONS**                                                                                                                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.0-01     | Page 19, Para  | KCOR \"does not attempt | Methodological Gap. Critics | Strengthen the         | High           |
|                | 2, Line 1 of   | to formally test\"      | may argue that without a    | \"Interpretability     |                |
|                | the section    | properties like frailty | formal goodness-of-fit test | Gates\" by defining a  |                |
|                |                | form.                   | (like a chi-square), the    | specific threshold for |                |
|                |                |                         | \"interpretability gates\"  | residual               |                |
|                |                |                         | are subjective.             | autocorrelation or fit |                |
|                |                |                         |                             | error that triggers a  |                |
|                |                |                         |                             | \"Non-Identified\"     |                |
|                |                |                         |                             | status.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.0-02     | Page 19, Item  | The observation of zero | This is a vital defense. If | Reinforce this by      | Low            |
|                | 3 of the list  | frailty (theta) in      | reviewers think you are     | stating that KCOR is   |                |
|                |                | vaccinated cohorts is   | assuming vaccinated people  | capable of fitting     |                |
|                |                | \"data-derived\".       | are \"perfectly equal,\"    | non-zero frailty to    |                |
|                |                |                         | they will reject the        | vaccinated cohorts if  |                |
|                |                |                         | premise.                    | the data curvature     |                |
|                |                |                         |                             | supports it.           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.0-03     | Page 19, Item  | \"Contamination of      | If an epidemic wave hits    | Advise users to        | High           |
|                | 5 of the list  | quiet periods\" by      | during the \"Quiet          | perform a \"Window     |                |
|                |                | external shocks.        | Window,\" the model will    | Shift\" sensitivity    |                |
|                |                |                         | treat that wave as          | analysis to ensure the |                |
|                |                |                         | \"depletion,\" leading to   | frailty estimate is    |                |
|                |                |                         | massive under-correction.   | stable across slightly |                |
|                |                |                         |                             | different quiet        |                |
|                |                |                         |                             | periods.               |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.0-04     | Page 19, Item  | Extension to non-fatal  | Competing Risk Risk. Unlike | Add a cautionary note  | Medium         |
|                | 6 of the list, | outcomes like           | death, hospitalization is   | that for recurrent     |                |
|                | Line 4         | hospitalization.        | not always terminal or      | events, KCOR should    |                |
|                |                |                         | irreversible. The risk set  | only be applied to the |                |
|                |                |                         | math changes significantly  | \"First Event\" to     |                |
|                |                |                         | for recurrent events.       | maintain the integrity |                |
|                |                |                         |                             | of the depletion       |                |
|                |                |                         |                             | logic.                 |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.0-05     | Page 20, 2^nd^ | \"Empirically, KCOR's   | This is your strongest      | Keep this phrasing     | Low            |
|                | last line of   | validity depends on     | argument for robustness. It | prominent; it          |                |
|                | the section    | curvature removal       | suggests the Gamma          | addresses the most     |                |
|                |                | rather than the         | distribution is a           | common objection to    |                |
|                |                | specific parametric     | \"Geometric Tool\" rather   | frailty models in one  |                |
|                |                | form\".                 | than a \"Biological         | sentence.              |                |
|                |                |                         | Claim\".                    |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **5.1 Failure Modes and Diagnostics**                                                                                             |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.1-01     | Page 20, Para  | External hazards (COVID | Major Interaction Risk. If  | Explicitly suggest     | High           |
|                | 2, Item 2 of   | waves) \"masquerading\" | a COVID wave hits, it       | that during            |                |
|                | the section    | as depletion.           | essentially \"speeds up\"   | high-intensity         |                |
|                |                |                         | the depletion process. If   | epidemic waves, the    |                |
|                |                |                         | your model doesn\'t account | \"Quiet Window\"       |                |
|                |                |                         | for this surge, it will     | assumption is likely   |                |
|                |                |                         | misestimate the underlying  | violated and KCOR      |                |
|                |                |                         | frailty.                    | should be used with    |                |
|                |                |                         |                             | extreme caution or not |                |
|                |                |                         |                             | at all.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.1-02     | Page 20, Para  | Weak identification in  | In rare-disease or          | Define a \"Minimum     | Medium         |
|                | 2, Item 3      | sparse cohorts.         | small-population studies,   | Event Count\"          |                |
|                |                |                         | the \"curvature\" might be  | threshold (e.g., at    |                |
|                |                |                         | just statistical noise      | least X events per     |                |
|                |                |                         | rather than a selection     | bin) required before   |                |
|                |                |                         | signal.                     | the frailty variance   |                |
|                |                |                         |                             | estimate can be        |                |
|                |                |                         |                             | considered stable.     |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.1-03     | Page 20, Para  | Use of \"H-space\"      | Cumulative hazards (H)      | Recommend viewing both | Medium         |
|                | 3, Item 2      | residuals (RMSE) as a   | always increase, which can  | H-space and h-space    |                |
|                |                | diagnostic.             | hide small but systematic   | residuals to ensure    |                |
|                |                |                         | misfits. A reviewer might   | that the \"local\" fit |                |
|                |                |                         | prefer looking at residuals | during the quiet       |                |
|                |                |                         | in \"h-space\"              | window is not masking  |                |
|                |                |                         | (instantaneous hazard).     | a \"global\" mismatch. |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.1-04     | Page 20, Para  | Stability under \"small | This is the gold standard   | Suggest a standard     | Low            |
|                | 3, Item 2      | perturbations\" of the  | for parameter stability. If | \"Stability Plot\"     |                |
|                |                | quiet-window bounds.    | moving the window by one    | (Theta vs. Window      |                |
|                |                |                         | week changes the result     | Start/End) as a        |                |
|                |                |                         | significantly, the model is | required diagnostic    |                |
|                |                |                         | not identified.             | for all KCOR reports.  |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **5.2 Conservativeness and Edge-Case Detection Limits**                                                                           |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.2-01     | Page 20, Para  | Treatment crossover     | Major Real-World            | Explicitly suggest a   | High           |
|                | 1, Line 2      | (unvaccinated getting   | Constraint. In long-term    | \"Crossover            |                |
|                |                | vaccinated) biases KCOR | studies, crossover is       | Threshold\" (e.g.,     |                |
|                |                | toward unity.           | inevitable. If the model    | stop analysis if \>10% |                |
|                |                |                         | always drifts toward 1.0    | of the cohort crosses  |                |
|                |                |                         | (no effect), it might mask  | over) to maintain the  |                |
|                |                |                         | a real vaccine benefit as   | integrity of the       |                |
|                |                |                         | time goes on.               | comparison.            |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.2-02     | Page 20, Para  | Absence of \"dose       | This is a clever            | Elevate this point. It | Low            |
|                | 2              | reversals\" as a        | \"Dose-Response\" check. If | shows that KCOR        |                |
|                |                | validation signal.      | Dose 2 looks riskier than   | results are logically  |                |
|                |                |                         | Dose 1 under the null, the  | consistent across      |                |
|                |                |                         | model is failing. Its       | different              |                |
|                |                |                         | absence in the Czech data   | \"strengths\" of       |                |
|                |                |                         | is a powerful endorsement.  | intervention.          |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.2-03     | Page 21, Line  | Overlap between         | Major Identification Risk.  | Reiterate that the     | High           |
|                | 6              | treatment effect and    | If the vaccine starts       | \"Quiet Window\" must  |                |
|                |                | quiet window leads to   | working during your \"Quiet | be justified by        |                |
|                |                | \"attenuation toward    | Window,\" you will          | external data showing  |                |
|                |                | unity\".                | accidentally normalize out  | a lack of intervention |                |
|                |                |                         | the vaccine\'s effect,      | activity (e.g.,        |                |
|                |                |                         | making it look useless.     | pre-peak epidemic).    |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.2-04     | Page 21, Para  | Acute effects in the    | If a vaccine causes rare    | Emphasize that KCOR is | Medium         |
|                | 2, Line 2 of   | \"Skip Window\" will    | but immediate adverse       | a tool for \"Sustained |                |
|                | the page       | not be captured.        | events (within 7 days),     | Effect\" and must be   |                |
|                |                |                         | KCOR will ignore them       | paired with a separate |                |
|                |                |                         | because they fall in the    | acute-safety analysis. |                |
|                |                |                         | skip period.                |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.2-05     | Page 20-21,    | Critique of Cox         | You are arguing that a Cox  | This is your strongest | Low            |
|                | General        | model-selection         | model can have a \"perfect  | technical \"mic        |                |
|                |                | criteria (AIC/BIC).     | fit\" but still be          | drop.\" Keep this as   |                |
|                |                |                         | \"perfectly wrong\" about   | the closing argument   |                |
|                |                |                         | the cumulative risk due to  | for why more complex   |                |
|                |                |                         | estimand mismatch.          | Cox models aren\'t the |                |
|                |                |                         |                             | answer.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **5.3 Data Requirements and External Validation**                                                                                 |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.3-01     | Page 21, Para  | Cohorts of 5,000 per    | Major Practical Guidance.   | Clarify if the         | High           |
|                | 1, Line 2 of   | arm yielded stable      | This is the first time a    | 5,000-person threshold |                |
|                | the section    | estimates in            | specific \"power\"          | assumes a certain      |                |
|                |                | simulation.             | threshold is mentioned. A   | baseline mortality     |                |
|                |                |                         | reviewer will want to know  | rate. Stability is     |                |
|                |                |                         | if this 5,000 refers to the | driven by the event    |                |
|                |                |                         | total population or the     | count, not just the    |                |
|                |                |                         | number of events.           | cohort size.           |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.3-02     | Page 21, Para  | RCTs are often          | This explains why KCOR is   | State that KCOR fills  | Low            |
|                | 2, Line 2 of   | \"underpowered\" and    | primarily a tool for        | the gap where RCTs     |                |
|                | the section    | lack record-level       | Observational Registries    | cannot go (e.g.,       |                |
|                |                | timing.                 | rather than clinical        | long-term all-cause    |                |
|                |                |                         | trials. It positions KCOR   | mortality in the       |                |
|                |                |                         | as the \"gold standard\"    | general population).   |                |
|                |                |                         | for real-world evidence     |                        |                |
|                |                |                         | (RWE).                      |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.3-03     | Page 21, Para  | Need for \"record-level | KCOR requires               | Add a \"Data           | Medium         |
|                | 2, Line 3 of   | timing\" or             | high-resolution temporal    | Resolution\"           |                |
|                | the section    | \"integrated health     | data (weekly or daily       | requirement: specify   |                |
|                |                | systems\".              | counts). If a registry only | that event binning     |                |
|                |                |                         | provides monthly or         | must be frequent       |                |
|                |                |                         | quarterly data, KCOR's      | enough to capture the  |                |
|                |                |                         | depletion math will fail.   | slope of the depletion |                |
|                |                |                         |                             | curve.                 |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| LIM-5.3-04     | Page 21,       | Cross-intervention      | This is a clever way to     | Frame this as the      | Medium         |
|                | General        | comparisons as a        | prove the method. If KCOR   | \"Population-Specific  |                |
|                |                | validation strategy.    | finds the same \"selection  | Frailty Signature\" to |                |
|                |                |                         | signal\" in a flu vaccine   | show that KCOR         |                |
|                |                |                         | cohort as it does in a      | captures an inherent   |                |
|                |                |                         | COVID vaccine cohort, the   | property of the        |                |
|                |                |                         | signal is a property of the | registry.              |                |
|                |                |                         | population, not the drug.   |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **6. CONCLUSION**                                                                                                                 |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| CON-6.0-02     | Page 21, Line  | Focus on \"Inverting    | Major Theoretical Claim.    | Use the term           | Medium         |
|                | 2 of the       | frailty-driven          | This implies KCOR isn\'t    | \"Restorative          |                |
|                | section        | depletion\".            | just a statistical          | Estimand\" to          |                |
|                |                |                         | adjustment, but a           | differentiate KCOR     |                |
|                |                |                         | mathematical restoration of | from \"Conditioning    |                |
|                |                |                         | the underlying baseline     | Estimands\" like Cox.  |                |
|                |                |                         | risk.                       |                        |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| CON-6.0-01     | Page 21, Line  | Claim that KCOR         | This is the strongest       | Ensure this is         | Low            |
|                | 3 of the       | \"remains near-null     | \"Selling Point.\" It       | explicitly linked back |                |
|                | section        | under selection without | directly answers the        | to the \"Synthetic     |                |
|                |                | effect\".               | failure of the Cox model    | Null\" and             |                |
|                |                |                         | shown in Figure 2.          | \"Age-Shift\"          |                |
|                |                |                         |                             | experiments to remind  |                |
|                |                |                         |                             | the reader of the      |                |
|                |                |                         |                             | evidence base.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| CON-6.0-03     | Page 21, Line  | \"Rather than presuming | This is the most important  | Consider adding one    | Low            |
|                | 3-4 of the     | identifiability, KCOR   | sentence for a skeptical    | final sentence stating |                |
|                | section        | enforces its            | reviewer. It changes the    | that KCOR provides a   |                |
|                |                | assumptions             | \"burden of proof\" from    | \"Fail-Safe\" for      |                |
|                |                | diagnostically\".       | the researcher to the data. | retrospective analysis |                |
|                |                |                         |                             | of registry data.      |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| **KCOR Supplementary Information (SI)**                                                                                           |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| SUP-S2-01      | Table S1--S3   | The \"Conservative      | This is a high-integrity    | Explicitly suggest     | Medium         |
|                |                | Failure Rule\" (I5)     | design choice. However, in  | that researchers       |                |
|                |                | mandates that results   | practice, it creates a      | report the percentage  |                |
|                |                | are not reported if     | \"Reporting Bias\" where    | of attempted cohorts   |                |
|                |                | diagnostics fail.       | only \"well-behaved\" data  | that failed            |                |
|                |                |                         | is published, potentially   | diagnostics to provide |                |
|                |                |                         | hiding regimes where KCOR   | context on the         |                |
|                |                |                         | is unusable.                | method\'s              |                |
|                |                |                         |                             | generalizability in a  |                |
|                |                |                         |                             | given dataset.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| SUP-S3-01      | Section S3.1   | Positive controls use a | Real-world treatment        | Briefly mention in the | Medium         |
|                |                | \"constant hazard       | effects (like vaccine       | SI if KCOR can recover |                |
|                |                | multiplier\" (r) over a | waning) are rarely          | a \"ramp\" or          |                |
|                |                | fixed interval.         | constant; they ramp up and  | \"decay\" effect       |                |
|                |                |                         | decay. A constant r is an   | shape, or if the       |                |
|                |                |                         | easy target for recovery.   | \"step function\" is   |                |
|                |                |                         |                             | the only shape tested. |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| SUP-S4-01      | Table S4       | Use of a                | This proves KCOR works even | Ensure the main text   | Low            |
|                |                | \"pathological\"        | when the population is      | (Section 3.1.1)        |                |
|                |                | frailty mixture with 5  | extremely fragmented. It is | cross-references this  |                |
|                |                | groups and weights.     | a powerful counter-argument | specific weight-shift  |                |
|                |                |                         | to the \"Gamma-only\"       | simulation to prove    |                |
|                |                |                         | critique.                   | the model handles      |                |
|                |                |                         |                             | non-smooth             |                |
|                |                |                         |                             | heterogeneity.         |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| SUP-S4.5-01    | Section S4.5   | Adversarial             | Major Stress Test. This is  | Highlight the          | High           |
|                |                | \"Tail-sampling\" where | the \"Bimodal\" case where  | \"non-identifiability  |                |
|                |                | frailty is concentrated | the mean is preserved but   | rate\" for this        |                |
|                |                | in the 0--15th and      | the variance is extreme.    | specific case. If KCOR |                |
|                |                | 85th--100th             | Success here is the         | identifies it, it\'s a |                |
|                |                | percentiles.            | ultimate proof of           | win; if it rejects it, |                |
|                |                |                         | \"Geometric Robustness\".   | the diagnostics are    |                |
|                |                |                         |                             | working as intended.   |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+
| SUP-S5.1-01    | Section S5.1   | RMSE_Hobs \< 0.01       | In large registries (N \>   | Emphasize the Residual | Medium         |
|                |                | threshold for           | 1M), even a \"good\" fit    | Analysis (S5.2) over   |                |
|                |                | \"excellent fit\".      | can have systematic         | the raw RMSE value, as |                |
|                |                |                         | residuals that are          | autocorrelation in     |                |
|                |                |                         | statistically significant   | residuals is a more    |                |
|                |                |                         | but numerically small.      | sensitive detector of  |                |
|                |                |                         |                             | model misspecification |                |
|                |                |                         |                             | than a single error    |                |
|                |                |                         |                             | metric.                |                |
+----------------+----------------+-------------------------+-----------------------------+------------------------+----------------+

: Add Inventory number, Item Description, Purchase Price, Quantity, and
Location to create an inventory list

+-----------------------------------------------------------------------+
| **General Comments**                                                  |
+:======================================================================+
| **Abstract**                                                          |
|                                                                       |
| Reviewers in observational epidemiology are trained to hunt for the   |
| word \"unbiased.\" If you use it, you must qualify it (e.g.,          |
| \"bias-reduced\" or \"unbiased in the absence of external             |
| confounding\"). Regarding the \"Recovering the true hazard ratio,\"   |
| This implies you have solved the \"fundamental problem of causal      |
| inference.\" Frame it instead as \"neutralizing a specific,           |
| measurable source of selection bias.\"                                |
|                                                                       |
| **Introduction**                                                      |
|                                                                       |
| **Section 1.1:** In the first few paragraphs, you are defining the    |
| \"enemy\" your method is fighting. If the reviewer thinks you are     |
| just describing \"unmeasured confounding,\" they will ask why you     |
| aren\'t using standard sensitivity analyses (like E-values). You must |
| ensure this section clearly differentiates static baseline            |
| confounding from dynamic, selection-induced frailty depletion. The    |
| \"Reviewer Risk\" is that a biostatistician will interpret your       |
| \"non-exchangeability\" description as a call for better propensity   |
| score matching, rather than a call for your new KCOR normalization.   |
|                                                                       |
| **Section 1.2:** You are using the \"linearity of the adjusted        |
| cumulative hazard\" as proof that the model worked. A skeptical       |
| reviewer will call this circular reasoning: *\"You assumed a          |
| gamma-frailty model, adjusted the data using that model, and then     |
| pointed to the fact that it now looks linear as proof the model is    |
| correct.\"* To mitigate this, you should emphasize that the \"quiet   |
| period\" (where linearity is expected) is separate from the \"active  |
| period\" where the actual effect is measured. This separation breaks  |
| the circularity.                                                      |
|                                                                       |
| **Methods**                                                           |
|                                                                       |
| **Section 2.1:** The distinction between Level and Curvature is where |
| you are trying to teach the reviewer a new way to look at survival    |
| data. As a reviewer, I will look for \"The Identifiability Warning.\" |
| If you have two cohorts, and Cohort A has a higher hazard than Cohort |
| B, is that because:                                                   |
|                                                                       |
| 1.  The intervention is bad? (Level)                                  |
|                                                                       |
| 2.  Or because Cohort A is \"frailer\" and depleting faster?          |
|     (Curvature)                                                       |
|                                                                       |
| Your \"Strategy\" claims you can separate these. You must ensure that |
| h_0(t) (the baseline hazard) and theta (the frailty) are not          |
| collinear. If they are collinear, you can\'t tell them apart, and     |
| KCOR is just \"overfitting the curve.\"                               |
|                                                                       |
| **Section 2.2:** The \"No Switching\" (MET-2.2-03) rule is your most  |
| controversial stance. The Critic\'s View: *\"If I enroll in the       |
| \'unvaccinated\' cohort but get vaccinated 4 weeks later, KCOR still  |
| treats me as \'unvaccinated.\' This is exposure misclassification.\"* |
|                                                                       |
| **Your Best Defense**: You have already started building this defense |
| by stating that transitions \"alter the frailty composition.\" You    |
| should lean into this: If you move the \"healthiest\" unvaccinated    |
| people into the vaccinated group mid-study, you have just created a   |
| massive, artificial depletion spike in the unvaccinated group that no |
| model can fix. KCOR prioritizes protecting the \"Selection Signal\"   |
| over \"Exposure Precision.\"                                          |
+-----------------------------------------------------------------------+
| **General Comments**                                                  |
+-----------------------------------------------------------------------+
| **Section 2.3:** If I am a reviewer and I see a paper claiming \"no   |
| loss to follow-up\" in a massive registry study, my \"red alert\"     |
| goes off. Registry data is inherently \"leaky.\" **The Fix:** You     |
| don\'t necessarily need to change the math, but you must change the   |
| defense. Instead of saying \"there is no loss to follow-up,\" say:    |
| \"For the purposes of the KCOR normalization, we treat the risk set   |
| as closed; sensitivity analyses in §5.2 demonstrate that the          |
| estimator is robust to non-informative censoring levels typical of    |
| national registries (e.g., \<1% per month).\" This moves it from a    |
| \"naive assumption\" to a \"tested design choice.\"                   |
|                                                                       |
| **Section 2.6:** **The Critic\'s View:** \"You found a \'correction   |
| factor\' in a quiet period of 2022. Why should I believe that same    |
| factor applies during a massive epidemic wave in 2023? The people     |
| dying in 2023 might be a totally different \'frailty slice\' than     |
| those in 2022.\"                                                      |
|                                                                       |
| **Your Best Defense:** You should argue that the Frailty Variance is  |
| a property of how the cohorts were selected and matched at the start. |
| It represents the \"hidden diversity\" of the group. Unless the group |
| is being replenished (which your \"fixed cohort\" rule prevents),     |
| that diversity is a structural constant. The inversion \"undoes\" the |
| mathematical consequence of that diversity so we can see the actual   |
| hazard spikes caused by external events.                              |
|                                                                       |
| **Section 2.11.1:** **The Critic's View:** \"You are showing that Cox |
| fails, but you picked a data-generating process (Gamma frailty) that  |
| exactly matches your KCOR model. Of course, KCOR wins!\" **Your Best  |
| Defense:** You should acknowledge that while the simulation uses the  |
| Gamma form, the Cox model fails regardless of the distribution        |
| because it cannot account for the changing composition of the risk    |
| set over time. The \"Major\" takeaway is that KCOR correctly          |
| identifies the \"zero effect\" because it operates on the neutralized |
| scale rather than the \"contaminated\" instantaneous scale.           |
|                                                                       |
| **Section 3.1.1:** **The Critic's View:** \"You have shown that if    |
| the world is exactly like your model, your model works. This is       |
| expected. The real question is what happens when the \'quiet window\' |
| is not perfectly quiet or the frailty is not Gamma-distributed.\"     |
| **Your Best Defense:** You are using this section to establish a      |
| \"Baseline of Trust\". By showing that the KCOR pipeline recovers the |
| truth under perfect conditions, you prove that the subsequent         |
| \"Real-World\" deviations are due to data complexity, not             |
| mathematical error in the estimator itself.                           |
|                                                                       |
| **Section 3.2:** **The Critic's View:** \"You spent several pages     |
| showing me that KCOR returns a result of 1.0 when the truth is 1.0.   |
| Now you tell me it can see real effects, but you\'ve hidden that      |
| proof in the Appendix. This makes me wonder if the \'neutralization\' |
| is so aggressive that it dampens real signals or makes them hard to   |
| interpret.\" **Your Best Defense:** You should argue that KCOR is     |
| specifically designed to preserve true hazards while removing         |
| selection artifacts. By moving a \"Positive Control\" figure into the |
| main text, you provide the necessary counter-weight to the negative   |
| controls. It shows the reader that the \"Neutralization Operator\" is |
| a surgical tool, not a blunt instrument that flattens all variation.  |
|                                                                       |
| **Overall Observation**                                               |
|                                                                       |
| **The Strength:** The \"Smoking Gun\" evidence in Section 2.11.1 and  |
| the empirical \"Age-Shift\" controls in Section 3.1.2 provide a level |
| of validation rarely seen in new survival estimators.                 |
|                                                                       |
| **The Guardrail:** The \"Interpretability Gates\" and \"Quiet         |
| Window\" logic ensure that the model fails gracefully rather than     |
| providing misleadingly significant p-values.                          |
+-----------------------------------------------------------------------+

: Signature layout table
