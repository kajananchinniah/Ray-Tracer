option(BUILD_DOCUMENTATION "Generate doxygen documentation" ON)

if(BUILD_DOCUMENTATION)
  find_package(Doxygen REQUIRED dot)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_GENERATE_MAN YES)

  set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs)
  doxygen_add_docs(doxygen-docs ${PROJECT_SOURCE_DIR})
endif()