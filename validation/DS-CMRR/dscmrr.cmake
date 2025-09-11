set(DATASET "${DATASET}" CACHE STRING "Dataset namespace")
set(PYTHON ${Python3_EXECUTABLE})
set(SCRIPT "${CMAKE_CURRENT_LIST_DIR}/DS-CMRR.py")
set(INPUT "${CMAKE_CURRENT_LIST_DIR}/../../data/${DATASET}/KCOR_CMR.xlsx")
set(OUTDIR "${CMAKE_CURRENT_LIST_DIR}/out")
set(OUTPUT "${OUTDIR}/DS-CMRR_output.xlsx")
set(LOG "DS_CMRR.log")

add_custom_command(
  OUTPUT ${OUTPUT}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTDIR}
  COMMAND ${PYTHON} ${SCRIPT} ${INPUT} ${OUTPUT} --log ${LOG} --balance-report --export-dose-pairs --sheet 2021_24 --by-age --dose-num-min 2 --dose-den-exact 0 --variance-method kish --save-png
  DEPENDS ${SCRIPT} ${INPUT}
  COMMENT "Running DS-CMRR validation"
)

add_custom_target(validation_DSCMRR DEPENDS ${OUTPUT})

add_custom_command(
  OUTPUT ${OUTDIR}/.by_age_only.stamp
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTDIR}
  COMMAND ${PYTHON} ${SCRIPT} ${INPUT} ${OUTPUT} --log ${LOG} --sheet 2021_24 --by-age-only --dose-num-min 2 --dose-den-exact 0 --variance-method kish
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTDIR}/.by_age_only.stamp
  DEPENDS ${SCRIPT} ${INPUT}
  COMMENT "Running DS-CMRR by-age-only"
)
add_custom_target(validation_DSCMRR_by_age_only DEPENDS ${OUTDIR}/.by_age_only.stamp)

add_custom_target(validation_DSCMRR_clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${OUTDIR} ${LOG}
)
