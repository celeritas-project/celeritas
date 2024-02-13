#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_add_library(testcel_core
  ScopedLogStorer.cc
)
celeritas_target_link_libraries(testcel_core
  PRIVATE Celeritas::testcel_harness Celeritas::corecel
)

celeritas_setup_tests(SERIAL PREFIX corecel
  LINK_LIBRARIES testcel_core Celeritas::corecel
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#
celeritas_add_test(OpaqueId.test.cc)

# Cont
set(CELERITASTEST_PREFIX corecel/cont)
celeritas_add_test(cont/Array.test.cc)
celeritas_add_test(cont/InitializedValue.test.cc)
celeritas_add_test(cont/Span.test.cc)
celeritas_add_test(cont/LabelIdMultiMap.test.cc)
celeritas_add_test(cont/VariantUtils.test.cc)
celeritas_add_device_test(cont/Range)

# Data
set(CELERITASTEST_PREFIX corecel/data)
celeritas_add_device_test(data/Collection)
celeritas_add_test(data/Copier.test.cc GPU)
celeritas_add_test(data/DeviceAllocation.test.cc GPU)
celeritas_add_test(data/DeviceVector.test.cc GPU)
celeritas_add_device_test(data/ObserverPtr)
celeritas_add_test(data/LdgIterator.test.cc)
celeritas_add_test(data/HyperslabIndexer.test.cc)
celeritas_add_device_test(data/StackAllocator)

# Grid
set(CELERITASTEST_PREFIX corecel/grid)
celeritas_add_test(grid/Interpolator.test.cc)
celeritas_add_test(grid/NonuniformGrid.test.cc)
celeritas_add_test(grid/TwodGridCalculator.test.cc)
celeritas_add_test(grid/UniformGrid.test.cc)

# IO
set(CELERITASTEST_PREFIX corecel/io)
celeritas_add_test(io/EnumStringMapper.test.cc)
celeritas_add_test(io/Label.test.cc)
celeritas_add_test(io/Join.test.cc)
celeritas_add_test(io/Logger.test.cc)
celeritas_add_test(io/OutputRegistry.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(io/Repr.test.cc)
celeritas_add_test(io/StringEnumMapper.test.cc)
celeritas_add_test(io/StringUtils.test.cc)

# Math
set(CELERITASTEST_PREFIX corecel/math)
celeritas_add_test(math/Algorithms.test.cc)
celeritas_add_test(math/ArrayOperators.test.cc)
celeritas_add_test(math/ArrayUtils.test.cc)
celeritas_add_test(math/HashUtils.test.cc)
celeritas_add_device_test(math/NumericLimits)
celeritas_add_test(math/Quantity.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(math/SoftEqual.test.cc)

# Sys
set(CELERITASTEST_PREFIX corecel/sys)
celeritas_add_test(sys/Environment.test.cc
  ENVIRONMENT "ENVTEST_ONE=1;ENVTEST_ZERO=0;ENVTEST_EMPTY="
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES}
)
celeritas_add_test(sys/MpiCommunicator.test.cc
  NP ${CELERITASTEST_NP_DEFAULT})
celeritas_add_test(sys/MultiExceptionHandler.test.cc)
celeritas_add_test(sys/TypeDemangler.test.cc)
celeritas_add_test(sys/ScopedSignalHandler.test.cc)
celeritas_add_test(sys/ScopedStreamRedirect.test.cc)
celeritas_add_test(sys/Stopwatch.test.cc ADDED_TESTS _stopwatch)
set_tests_properties(${_stopwatch} PROPERTIES LABELS "nomemcheck")
celeritas_add_test(sys/Version.test.cc)


#-----------------------------------------------------------------------------#
