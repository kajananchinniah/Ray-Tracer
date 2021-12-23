find_package(GTest REQUIRED)

set(COMMON_TEST_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(test_image_utils test_image_utils.cpp)
target_include_directories(test_image_utils PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_image_utils PRIVATE ${COMMON_TEST_LIBRARIES} raytracer_image_utils)
add_test(TestImageUtils test_image_utils ${CMAKE_CURRENT_LIST_DIR}/test_resources/123456.png)
