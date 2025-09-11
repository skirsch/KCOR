set(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(DATASET "${DATASET}" CACHE STRING "Dataset namespace")
set(SRC_CMR "${ROOT_DIR}/data/${DATASET}/KCOR_CMR.xlsx")
set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/data")
set(OUT_DIR "${CMAKE_CURRENT_LIST_DIR}/out")
set(CODE_DIR "${CMAKE_CURRENT_LIST_DIR}/code")
set(SHEETS "2021_24,2022_06" CACHE STRING "Sheets for negative control")
set(SYNTH "${DATA_DIR}/KCOR_synthetic_neg_control.xlsx")
set(OUT_XLSX "${OUT_DIR}/KCOR_processed_neg_control.xlsx")

add_custom_command(
  OUTPUT ${SYNTH}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DATA_DIR}
  COMMAND ${Python3_EXECUTABLE} ${CODE_DIR}/generate_negative_control.py ${SRC_CMR} ${SYNTH} ${SHEETS}
  DEPENDS ${CODE_DIR}/generate_negative_control.py ${SRC_CMR}
  COMMENT "Generating synthetic negative control"
)

add_custom_command(
  OUTPUT ${OUT_XLSX}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E env SA_FINAL_KCOR_MIN=0 ${Python3_EXECUTABLE} ${ROOT_DIR}/code/KCOR.py ${SYNTH} ${OUT_XLSX} KCOR_summary_neg_control.log
  DEPENDS ${ROOT_DIR}/code/KCOR.py ${SYNTH}
  COMMENT "Running negative control analysis"
)

add_custom_target(test_negative_control DEPENDS ${OUT_XLSX})

add_custom_target(test_negative_control_clean
  COMMAND ${CMAKE_COMMAND} -E rm -f ${SYNTH} ${OUT_XLSX} ${OUT_DIR}/KCOR_summary_neg_control.log
)
