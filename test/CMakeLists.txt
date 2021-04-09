add_executable(test_grid test_grid.cu)
add_test(NAME test_grid COMMAND test_grid)

add_executable(test_poisson test_poisson.cu)
add_test(NAME test_poisson COMMAND test_poisson)
