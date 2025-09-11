set(DATASET "${DATASET}" CACHE STRING "Dataset namespace")
set(PYTHON ${Python3_EXECUTABLE})
set(SCRIPT "${CMAKE_CURRENT_LIST_DIR}/code/KM.py")
set(INPUT "${CMAKE_CURRENT_LIST_DIR}/../../data/${DATASET}/KCOR_CMR.xlsx")
set(OUTDIR "${CMAKE_CURRENT_LIST_DIR}/out")

set(SHEET "2021_24" CACHE STRING "KM sheet")
set(START "1950" CACHE STRING "Start year")
set(END "1950" CACHE STRING "End year")
set(PLOT "0,2,1" CACHE STRING "Plot spec")
set(EQUAL_POPULATION "0" CACHE STRING "Equal population")
set(OUTPNG "${OUTDIR}/KM_${SHEET}_${START}_${END}.png")

add_custom_command(
  OUTPUT ${OUTPNG}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTDIR}
  COMMAND ${PYTHON} ${SCRIPT} --sheet ${SHEET} --birth-years ${START} ${END} --plot ${PLOT} --out ${OUTPNG} --input ${INPUT} --equal-population ${EQUAL_POPULATION}
  DEPENDS ${SCRIPT} ${INPUT}
  COMMENT "Running Kaplan–Meier validation"
)

add_custom_target(validation_kaplan_meier DEPENDS ${OUTPNG})

add_custom_target(validation_kaplan_meier_clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${OUTDIR}
)
