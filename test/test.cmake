include(${CMAKE_CURRENT_LIST_DIR}/negative_control/negative_control.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/sensitivity/sensitivity.cmake)

add_custom_target(test_all DEPENDS test_negative_control test_sensitivity)
add_custom_target(test_clean DEPENDS test_negative_control_clean test_sensitivity_clean)
