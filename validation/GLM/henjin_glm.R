# Henjin-style GLM on Czech NZIP variable cohorts
# Reconstructed from screenshot you provided
# Requires: data.table

library(data.table)

# Read data (gzip CSV from NZIP export)
t <- fread("Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv.gz")

# Keep rows with Infekce in {NA, 1} and valid birth year
t <- t[Infekce %in% c(NA, 1)][RokNarozeni != "-"]

# First-dose week and drop rows where death precedes dose1
t[, dose1 := Datum_Prvni_davka][dose1 > "2021-23", dose1 := ""]
t <- t[!(DatumUmrtiLPZ != "" & dose1 > DatumUmrtiLPZ)]

# Birth year numeric
t[, born := as.integer(substr(RokNarozeni, 1, 4))]

# Build per-week variable cohorts: unvax baseline "2020-10" and observed dose1 week
a <- t[, .(week = c(rep("2020-10", .N), dose1), dose = rep(0:1, each = .N), born)]
a <- a[, .(vax = .N), .(week, dose, born)]

# Add deaths by (week, dose, born)
a <- merge(a,
           t[, .(dead = .N), .(week = DatumUmrtiLPZ, dose = (dose1 != "") * 1, born)],
           all = TRUE)[week != ""]

# Fill implicit zero cells via Cartesian join over observed levels
a <- merge(do.call(CJ, lapply(a[, 1:3], unique)), a, all = TRUE); a[is.na(a)] <- 0

# Cumulative vaccination counts and population at risk by (dose, born)
a[, cumvax := cumsum(vax), .(dose, born)][, pop := cumvax - cumsum(dead), .(dose, born)]
a[dose == 0, pop := pop - a[dose == 1, cumvax]]

# Collapse within cells and restrict analysis window
a <- a[, .(dead = sum(dead), pop = sum(pop)), .(week, dose, born)]
a <- a[week >= "2021-24" & week <= "2024-26"]

# Make cumulative within (dose, born) over calendar time
a[, c("dead","pop") := .(cumsum(dead), cumsum(pop)), .(dose, born)]

# Baseline "base" level and factor releveling
a <- rbind(a, a[week == "2021-34"][, week := "base"])
a[, week := relevel(factor(week), "base")]

# Poisson GLM with offset(log(pop)), age adjustment, dose×week interaction
fit <- glm(dead ~ dose * week + factor(born), poisson, a, offset = log(pop))

# Extract weekly dose effects relative to baseline
p <- CJ(week = levels(a$week)[-1])
i <- match(paste0("dose:week", p$week), names(coef(fit)))
est <- exp(coef(fit)[i])
se  <- sqrt(diag(vcov(fit)))[i]

# Build table with y and 95% CI (lo, y, hi)
p <- cbind(p, `colnames<-`(est + outer(qnorm(.975) * se, -1:1), c("lo","y","hi")))

# Print head for convenience
print(head(p))
