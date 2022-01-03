find_package(GTest REQUIRED)

# Ignore clang tidy
set(CMAKE_CXX_CLANG_TIDY "")

set(COMMON_TEST_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

# Image Utils
add_executable(test_image_utils test_image_utils.cpp)
target_include_directories(test_image_utils PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_image_utils PRIVATE ${COMMON_TEST_LIBRARIES}
                                               raytracer_common raytracer_image)
add_test(TestImageUtils test_image_utils
         ${CMAKE_CURRENT_LIST_DIR}/test_resources/123456.png)

# Vector3
add_executable(test_vector3 test_vector3.cpp)
target_include_directories(test_vector3 PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_vector3 PRIVATE ${COMMON_TEST_LIBRARIES}
                                           raytracer_common raytracer_vector3)
add_test(TestVector3 test_vector3)

# Rendering
add_executable(test_render test_render.cu)
target_include_directories(test_render PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(
  test_render
  PRIVATE ${COMMON_TEST_LIBRARIES}
          raytracer_image
          raytracer_common
          raytracer_vector3
          raytracer_ray
          raytracer_camera
          raytracer_material)
add_test(TestRender test_render ${CMAKE_CURRENT_LIST_DIR}/test_resources)
set_target_properties(test_render PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Renable clang tidy
if(CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy-10")
endif()
