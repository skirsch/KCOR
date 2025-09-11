# Dataset configuration propagated from parent
set(DATASET "${DATASET}" CACHE STRING "Dataset namespace")

set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../data/${DATASET}")
set(RAW_CSV_GZ "${DATA_DIR}/Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv.gzip")
set(RAW_CSV "${DATA_DIR}/Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv")
set(CMR_OUTPUT "${DATA_DIR}/KCOR_CMR.xlsx")
set(KCOR_OUTPUT "${DATA_DIR}/KCOR.xlsx")
set(VARIABLE_OUTPUT "${DATA_DIR}/KCOR_variable.xlsx")

# Inflate raw CSV from gzip asset
add_custom_command(
  OUTPUT ${RAW_CSV}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/decompress_gzip.py ${RAW_CSV_GZ} ${RAW_CSV}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/decompress_gzip.py ${RAW_CSV_GZ}
  COMMENT "Decompressing raw CSV"
)

# CMR aggregation step
add_custom_command(
  OUTPUT ${CMR_OUTPUT}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/KCOR_CMR.py ${RAW_CSV} ${CMR_OUTPUT}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/KCOR_CMR.py ${RAW_CSV}
  COMMENT "Building CMR output"
)

# KCOR analysis step
add_custom_command(
  OUTPUT ${KCOR_OUTPUT}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/KCOR.py ${CMR_OUTPUT} ${KCOR_OUTPUT} "Primary Analysis" KCOR_summary.log
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/KCOR.py ${CMR_OUTPUT}
  COMMENT "Running KCOR analysis"
)

# Variable-cohort aggregation
add_custom_command(
  OUTPUT ${VARIABLE_OUTPUT}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/KCOR_variable.py ${RAW_CSV} ${VARIABLE_OUTPUT}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/KCOR_variable.py ${RAW_CSV}
  COMMENT "Building variable cohort output"
)

add_custom_target(code_KCOR DEPENDS ${KCOR_OUTPUT})
add_custom_target(code_KCOR_variable DEPENDS ${VARIABLE_OUTPUT})

add_custom_target(code_clean
  COMMAND ${CMAKE_COMMAND} -E rm -f ${CMR_OUTPUT} ${KCOR_OUTPUT} ${VARIABLE_OUTPUT}
)
