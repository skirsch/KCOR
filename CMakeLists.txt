cmake_minimum_required(VERSION 3.15)
project(KCOR NONE)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(DATASET "Czech" CACHE STRING "Dataset namespace")

include(code/code.cmake)
include(test/test.cmake)
include(validation/DS-CMRR/dscmrr.cmake)
include(validation/kaplan_meier/kaplan_meier.cmake)
include(validation/GLM/glm.cmake)

add_custom_target(run DEPENDS code_KCOR)
add_custom_target(KCOR_variable DEPENDS code_KCOR_variable)
add_custom_target(validation DEPENDS validation_DSCMRR validation_kaplan_meier validation_GLM)
add_custom_target(test DEPENDS test_all)
add_dependencies(validation_GLM code_KCOR_variable)
add_custom_target(km DEPENDS validation_kaplan_meier)
add_custom_target(glm DEPENDS validation_GLM)
add_custom_target(glm_compare DEPENDS validation_GLM_compare)
add_custom_target(sensitivity DEPENDS test_sensitivity)
add_custom_target(kcor_all ALL DEPENDS KCOR_variable run validation test)
add_custom_target(kcor_clean DEPENDS code_clean validation_DSCMRR_clean validation_kaplan_meier_clean validation_GLM_clean test_clean)
