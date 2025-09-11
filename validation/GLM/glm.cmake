set(DATASET "${DATASET}" CACHE STRING "Dataset namespace")
set(PYTHON ${Python3_EXECUTABLE})
set(SCRIPT "${CMAKE_CURRENT_LIST_DIR}/code/GLM.py")
set(COMPARE_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/code/compare_pngs.py")
set(INPUT "${CMAKE_CURRENT_LIST_DIR}/../../data/${DATASET}/KCOR_variable.xlsx")
set(OUTDIR "${CMAKE_CURRENT_LIST_DIR}/out")
set(OUTPNG "${OUTDIR}/GLM_plot_Czech_data.png")
set(OUTXLSX "${OUTDIR}/GLM_output.xlsx")
set(REFPNG "${CMAKE_CURRENT_LIST_DIR}/../../reference_results/GLM/GLM_plot_Czech_data.png")

add_custom_command(
  OUTPUT ${OUTPNG}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTDIR}
  COMMAND ${PYTHON} ${SCRIPT} --input ${INPUT} --out-png ${OUTPNG} --out-xlsx ${OUTXLSX}
  DEPENDS ${SCRIPT} ${INPUT}
  COMMENT "Running GLM validation"
)

add_custom_target(validation_GLM DEPENDS ${OUTPNG})

add_custom_target(validation_GLM_clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${OUTDIR}
)

add_custom_command(
  OUTPUT ${OUTDIR}/GLM_compare.png ${OUTDIR}/GLM_compare.txt
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTDIR}
  COMMAND ${PYTHON} ${COMPARE_SCRIPT} --a ${OUTPNG} --b ${REFPNG} --out ${OUTDIR}/GLM_compare.png --report ${OUTDIR}/GLM_compare.txt
  DEPENDS ${COMPARE_SCRIPT} ${OUTPNG} ${REFPNG}
  COMMENT "Comparing GLM output with reference"
)
add_custom_target(validation_GLM_compare DEPENDS ${OUTDIR}/GLM_compare.png ${OUTDIR}/GLM_compare.txt)
