---
name: ""
overview: ""
todos: []
---

# Plan: Fix Remaining Reference Artifacts and Formatting Issues

## Issues to Fix

### 1. Equation Reference Prefixes

- **Problem**: References render as "Equation eq. 5" instead of "Equation 5" or "Equation (5)"
- **Location**: Line 982: `Equation @eq:gamma-frailty-identity`
- **Solution**: Update `pandoc-crossref.yaml` to remove equation prefix
- Set `eqnPrefix: ""` and update `eqnTemplate` to use just `($$i$$)` or `$$i$$`
- Since user supplies "Equation" themselves, template should just provide number

### 2. Truncated Heading (Verify)

- **Problem**: User reports heading shows as "2.1.3 Identifiability and sco" (truncated)
- **Location**: Line 232: `#### 2.1.3 Identifiability and scope of inference`
- **Status**: Heading appears complete in source markdown
- **Solution**: Verify heading is correct; check for hidden characters or line breaks

### 3. Stray "scope of inference" Line (Verify)

- **Problem**: User reports stray line "scope of inference" above section 2.1.3
- **Location**: Line 226 mentions "scope of inference" in text, line 232 has it in heading
- **Solution**: Check for formatting issues or duplicate text

## Files to Modify

- `documentation/preprint/pandoc-crossref.yaml` - Fix equation template
- `documentation/preprint/paper.md` - Verify/fix heading

## Approach

1. **Fix equation references**:

- Update `eqnTemplate` in `pandoc-crossref.yaml` to `"($$i$$)"` or `"$$i$$"`
- Set `eqnPrefix: ""` to remove prefix
- User supplies "Equation" themselves, so template just provides number

2. **Verify heading**:

- Check line 232 for any hidden characters or formatting issues
- Ensure heading is on single line without breaks

3. **Check for stray text**:

- Verify no duplicate or misplaced text around section 2.1.3

## Notes

- KCOR6_FIT is intentional log output - no changes needed