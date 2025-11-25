#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_add_test_library(testcel_core
  StringSimplifier.cc
  StringSimplifier.cc
  ScopedLogStorer.cc
  random/Histogram.cc
  random/HistogramSampler.cc
)
celeritas_target_link_libraries(testcel_core
  PRIVATE testcel_harness Celeritas::corecel
)

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES testcel_core Celeritas::corecel
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#
celeritas_add_test(StringSimplifier.test.cc)

celeritas_add_test(Assert.test.cc
  ENVIRONMENT "CELER_COLOR=1;CELER_LOG=debug"
)
celeritas_add_test(OpaqueId.test.cc)

# Cont
celeritas_add_test(cont/Array.test.cc)
celeritas_add_test(cont/Bitset.test.cc)
celeritas_add_test(cont/InitializedValue.test.cc)
celeritas_add_test(cont/Span.test.cc)
celeritas_add_test(cont/LabelIdMultiMap.test.cc)
celeritas_add_test(cont/VariantUtils.test.cc)
celeritas_add_device_test(cont/Range
  LINK_LIBRARIES Celeritas::ExtThrust)

# Data
celeritas_add_device_test(data/Collection)
celeritas_add_test(data/Copier.test.cc GPU)
celeritas_add_test(data/DeviceAllocation.test.cc GPU)
celeritas_add_test(data/DeviceVector.test.cc GPU)
celeritas_add_test(data/Filler.test.cc GPU)
celeritas_add_device_test(data/ObserverPtr
  LINK_LIBRARIES Celeritas::ExtThrust)
celeritas_add_test(data/LdgIterator.test.cc)
celeritas_add_test(data/HyperslabIndexer.test.cc)
celeritas_add_device_test(data/StackAllocator
  LINK_LIBRARIES Celeritas::ExtThrust)
celeritas_add_test(data/AuxInterface.test.cc
  SOURCES data/AuxMockParams.cc)

# Grid
celeritas_add_test(grid/FindInterp.test.cc)
celeritas_add_test(grid/Interpolator.test.cc)
celeritas_add_test(grid/NonuniformGrid.test.cc)
celeritas_add_test(grid/SplineDerivCalculator.test.cc)
celeritas_add_test(grid/SplineInterpolator.test.cc)
celeritas_add_test(grid/TwodGridCalculator.test.cc)
celeritas_add_test(grid/UniformGrid.test.cc)
celeritas_add_test(grid/VectorUtils.test.cc)

# IO
celeritas_add_test(io/EnumStringMapper.test.cc)
celeritas_add_test(io/Label.test.cc)
celeritas_add_test(io/Join.test.cc)
celeritas_add_test(io/Logger.test.cc
  LINK_LIBRARIES Celeritas::ExtMPI)
celeritas_add_test(io/OutputRegistry.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json)
celeritas_add_test(io/Repr.test.cc)
celeritas_add_test(io/ScopedStreamRedirect.test.cc)
celeritas_add_test(io/StringEnumMapper.test.cc)
celeritas_add_test(io/StringUtils.test.cc)

# Math
celeritas_add_device_test(math/Algorithms)
celeritas_add_test(math/ArrayOperators.test.cc)
celeritas_add_test(math/ArrayUtils.test.cc)
celeritas_add_test(math/PdfUtils.test.cc)
celeritas_add_test(math/Constant.test.cc)
celeritas_add_test(math/HashUtils.test.cc)
celeritas_add_test(math/Integrator.test.cc)
celeritas_add_device_test(math/NumericLimits)
celeritas_add_test(math/PolyEvaluator.test.cc)
celeritas_add_test(math/Quantity.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json)
celeritas_add_test(math/SoftEqual.test.cc)
celeritas_add_test(math/RegulaFalsiRootFinder.test.cc)
celeritas_add_test(math/BisectionRootFinder.test.cc)
celeritas_add_test(math/IllinoisRootFinder.test.cc)
celeritas_add_test(math/TridiagonalSolver.test.cc)

#-----------------------------------------------------------------------------#
# Random

celeritas_add_test(random/CachedRngEngine.test.cc)
celeritas_add_test(random/Histogram.test.cc)
celeritas_add_device_test(random/RngEngine
  LINK_LIBRARIES Celeritas::ExtThrust)
celeritas_add_test(random/Selector.test.cc)
celeritas_add_test(random/XorwowRngEngine.test.cc GPU)

celeritas_add_test(random/distribution/BernoulliDistribution.test.cc)
celeritas_add_test(random/distribution/ExponentialDistribution.test.cc)
celeritas_add_test(random/distribution/GammaDistribution.test.cc)
celeritas_add_test(random/distribution/InverseSquareDistribution.test.cc)
celeritas_add_test(random/distribution/IsotropicDistribution.test.cc)
celeritas_add_test(random/distribution/NormalDistribution.test.cc)
celeritas_add_test(random/distribution/PoissonDistribution.test.cc)
celeritas_add_test(random/distribution/PowerDistribution.test.cc)
celeritas_add_test(random/distribution/RadialDistribution.test.cc)
celeritas_add_test(random/distribution/ReciprocalDistribution.test.cc)
celeritas_add_test(random/distribution/RejectionSampler.test.cc)
celeritas_add_test(random/distribution/UniformBoxDistribution.test.cc)
celeritas_add_test(random/distribution/UniformRealDistribution.test.cc)

#-----------------------------------------------------------------------------#
# Sys
celeritas_add_test(sys/ActionRegistry.test.cc)
celeritas_add_test(sys/Device.test.cc GPU
  LINK_LIBRARIES nlohmann_json::nlohmann_json
)
celeritas_add_test(sys/Environment.test.cc
  ENVIRONMENT "ENVTEST_ONE=1;ENVTEST_ZERO=0;ENVTEST_EMPTY="
  LINK_LIBRARIES nlohmann_json::nlohmann_json
)
celeritas_add_test(sys/MpiCommunicator.test.cc
  LINK_LIBRARIES Celeritas::ExtMPI
  NP ${CELERITASTEST_NP_DEFAULT})
celeritas_add_test(sys/MultiExceptionHandler.test.cc)
celeritas_add_test(sys/TypeDemangler.test.cc)
celeritas_add_test(sys/ScopedSignalHandler.test.cc)
celeritas_add_test(sys/Stopwatch.test.cc ADDED_TESTS _stopwatch)
set_tests_properties(${_stopwatch} PROPERTIES LABELS "nomemcheck")
celeritas_add_test(sys/Version.test.cc)


#-----------------------------------------------------------------------------#
