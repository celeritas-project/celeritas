#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

include(CeleritasGen)

#----------------------------------------------------------------------------#
# CONFIGURE FILES
#----------------------------------------------------------------------------#
set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_OPENMP  ${CELERITAS_USE_OpenMP})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})

# Define a numeric table of options for the default runtime RNG.
# Start counter from 1 because undefined macros have the implicit value of 0 in
# the C preprocessor, so any unavailable options (e.g. CELERITAS_USE_CURAND
# when HIP is in use) will implicitly be zero.
set(CELERITAS_RNG_MACROS)
set(_rng_counter 1)
foreach(_rng IN LISTS CELERITAS_RNG_OPTIONS)
  list(APPEND CELERITAS_RNG_MACROS
    "#define CELERITAS_RNG_${_rng} ${_rng_counter}"
  )
  math(EXPR _rng_counter "${_rng_counter} + 1")
endforeach()

string(JOIN "\n" CELERITAS_RNG_MACROS
  ${CELERITAS_RNG_MACROS}
  "#define CELERITAS_RNG CELERITAS_RNG_${CELERITAS_RNG}"
)
configure_file("celeritas_config.h.in" "celeritas_config.h" @ONLY)

#----------------------------------------------------------------------------#

# TODO: add build flags
set(CELERITAS_CMAKE_STRINGS
    "static const char celeritas_rng[] = \"${CELERITAS_RNG}\";\n"
)
set(BUILD_TYPE ${CMAKE_BUILD_TYPE})
foreach(_var BUILD_TYPE CLHEP_VERSION Geant4_VERSION VecGeom_VERSION)
  string(TOLOWER "${_var}" _lower)
  string(APPEND CELERITAS_CMAKE_STRINGS
    "static const char celeritas_${_lower}[] = \"${${_var}}\";\n"
  )
endforeach()
configure_file("celeritas_cmake_strings.h.in" "celeritas_cmake_strings.h" @ONLY)

#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

set(SOURCES)
set(DEVICE_SOURCES)
set(PRIVATE_DEPS)
set(PUBLIC_DEPS)

# Version information
configure_file("celeritas_version.cc.in" "celeritas_version.cc" @ONLY)
list(APPEND SOURCES
  "${CMAKE_CURRENT_BINARY_DIR}/celeritas_version.cc"
)

# Generate model launcher and track initialization code for host/device
set(_gen_sources)
macro(celeritas_gen_interactor class func)
  celeritas_gen(_gen_sources
    "gen-interactor.py" "celeritas/em/generated/${class}Interact"
    --class ${class}  --func ${func}
  )
endmacro()
macro(celeritas_gen_action basedir class func)
  celeritas_gen(_gen_sources
    "gen-action.py" "celeritas/${basedir}/generated/${class}"
    --class ${class} --func ${func}
  )
endmacro()
macro(celeritas_gen_trackinit class)
  celeritas_gen(_gen_sources
    "gen-trackinit.py" "celeritas/track/generated/${class}"
    --class ${class}
  )
endmacro()

macro(celeritas_polysource filename_we)
  list(APPEND SOURCES "${filename_we}.cc")
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND DEVICE_SOURCES "${filename_we}.cu")
  endif()
endmacro()

celeritas_gen_interactor("BetheHeitler" "bethe_heitler")
celeritas_gen_interactor("CombinedBrem" "combined_brem")
celeritas_gen_interactor("EPlusGG" "eplusgg")
celeritas_gen_interactor("KleinNishina" "klein_nishina")
celeritas_gen_interactor("LivermorePE" "livermore_pe")
celeritas_gen_interactor("MollerBhabha" "moller_bhabha")
celeritas_gen_interactor("MuBremsstrahlung" "mu_bremsstrahlung")
celeritas_gen_interactor("Rayleigh" "rayleigh")
celeritas_gen_interactor("RelativisticBrem" "relativistic_brem")
celeritas_gen_interactor("SeltzerBerger" "seltzer_berger")

celeritas_gen_action("phys" "DiscreteSelectAction" "discrete_select")
celeritas_gen_action("phys" "PreStepAction" "pre_step")
celeritas_gen_action("geo" "BoundaryAction" "boundary")

celeritas_gen_trackinit("InitTracks")
celeritas_gen_trackinit("LocateAlive")
celeritas_gen_trackinit("ProcessPrimaries")
celeritas_gen_trackinit("ProcessSecondaries")

list(APPEND SOURCES ${_gen_sources})

# Main library
list(APPEND SOURCES
  corecel/Assert.cc
  corecel/cont/Label.cc
  corecel/data/Copier.cc
  corecel/data/DeviceAllocation.cc
  corecel/io/BuildOutput.cc
  corecel/io/ColorUtils.cc
  corecel/io/Logger.cc
  corecel/io/LoggerTypes.cc
  corecel/io/OutputInterface.cc
  corecel/io/OutputManager.cc
  corecel/io/ScopedStreamRedirect.cc
  corecel/io/ScopedTimeAndRedirect.cc
  corecel/io/StringUtils.cc
  corecel/io/detail/LoggerMessage.cc
  corecel/io/detail/ReprImpl.cc
  corecel/math/VectorUtils.cc
  corecel/sys/Device.cc
  corecel/sys/Environment.cc
  corecel/sys/KernelDiagnostics.cc
  corecel/sys/ScopedSignalHandler.cc
  corecel/sys/TypeDemangler.cc

  orange/OrangeParams.cc
  orange/Types.cc
  orange/construct/SurfaceInserter.cc
  orange/construct/VolumeInserter.cc
  orange/surf/SurfaceIO.cc

  celeritas/em/AtomicRelaxationParams.cc
  celeritas/em/FluctuationParams.cc
  celeritas/em/detail/Utils.cc
  celeritas/em/model/BetheHeitlerModel.cc
  celeritas/em/model/CombinedBremModel.cc
  celeritas/em/model/EPlusGGModel.cc
  celeritas/em/model/KleinNishinaModel.cc
  celeritas/em/model/LivermorePEModel.cc
  celeritas/em/model/MollerBhabhaModel.cc
  celeritas/em/model/MuBremsstrahlungModel.cc
  celeritas/em/model/RayleighModel.cc
  celeritas/em/model/RelativisticBremModel.cc
  celeritas/em/model/SeltzerBergerModel.cc
  celeritas/em/model/UrbanMscModel.cc
  celeritas/em/process/BremsstrahlungProcess.cc
  celeritas/em/process/ComptonProcess.cc
  celeritas/em/process/EIonizationProcess.cc
  celeritas/em/process/EPlusAnnihilationProcess.cc
  celeritas/em/process/GammaConversionProcess.cc
  celeritas/em/process/MultipleScatteringProcess.cc
  celeritas/em/process/PhotoelectricProcess.cc
  celeritas/em/process/RayleighProcess.cc
  celeritas/ext/MpiCommunicator.cc
  celeritas/ext/ScopedMpiInit.cc
  celeritas/geo/GeoMaterialParams.cc
  celeritas/global/ActionInterface.cc
  celeritas/global/ActionManager.cc
  celeritas/global/ActionManagerOutput.cc
  celeritas/global/CoreParams.cc
  celeritas/global/Stepper.cc
  celeritas/grid/ValueGridBuilder.cc
  celeritas/grid/ValueGridInserter.cc
  celeritas/grid/ValueGridInterface.cc
  celeritas/io/AtomicRelaxationReader.cc
  celeritas/io/ImportData.cc
  celeritas/io/ImportPhysicsTable.cc
  celeritas/io/ImportPhysicsVector.cc
  celeritas/io/ImportProcess.cc
  celeritas/io/LivermorePEReader.cc
  celeritas/io/SeltzerBergerReader.cc
  celeritas/mat/MaterialParams.cc
  celeritas/mat/detail/Utils.cc
  celeritas/phys/CutoffParams.cc
  celeritas/phys/ImportedModelAdapter.cc
  celeritas/phys/ImportedProcessAdapter.cc
  celeritas/phys/ParticleParams.cc
  celeritas/phys/PhysicsParams.cc
  celeritas/phys/PhysicsParamsOutput.cc
  celeritas/phys/PrimaryGenerator.cc
  celeritas/phys/Process.cc
  celeritas/phys/ProcessBuilder.cc
  celeritas/random/CuHipRngData.cc
  celeritas/random/XorwowRngData.cc
  celeritas/random/XorwowRngParams.cc
  celeritas/track/TrackInitParams.cc
)

celeritas_polysource(celeritas/global/alongstep/AlongStepGeneralLinearAction)
celeritas_polysource(celeritas/global/alongstep/AlongStepNeutralAction)
celeritas_polysource(celeritas/global/alongstep/AlongStepUniformMscAction)
celeritas_polysource(celeritas/random/detail/CuHipRngStateInit)
celeritas_polysource(celeritas/track/detail/TrackInitAlgorithms)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    corecel/sys/KernelParamCalculator.device.cc
  )
  list(APPEND DEVICE_SOURCES
    corecel/data/detail/Filler.cu
  )
endif()

if(CELERITAS_USE_HepMC3)
  list(APPEND SOURCES
    celeritas/io/EventReader.cc
  )
  list(APPEND PRIVATE_DEPS HepMC3::HepMC3)
else()
  list(APPEND SOURCES
    celeritas/io/EventReader.nohepmc.cc
  )
endif()

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    corecel/sys/DeviceIO.json.cc
    corecel/sys/KernelDiagnosticsIO.json.cc
    celeritas/phys/PrimaryGeneratorOptionsIO.json.cc
    orange/construct/SurfaceInputIO.json.cc
    orange/construct/VolumeInputIO.json.cc
    celeritas/field/FieldDriverOptionsIO.json.cc
  )
  list(APPEND PUBLIC_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_RNG STREQUAL "CURAND")
  list(APPEND PUBLIC_DEPS Celeritas::DeviceToolkit)
else()
  list(APPEND PRIVATE_DEPS Celeritas::DeviceToolkit)
endif()

if(CELERITAS_USE_MPI)
  list(APPEND PUBLIC_DEPS MPI::MPI_CXX)
endif()

if(CELERITAS_USE_OpenMP)
  list(APPEND PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    celeritas/ext/VecgeomParams.cc
    celeritas/ext/detail/VecgeomNavCollection.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml)
  # This needs to be public because its might be needed
  # to resolve the symbol generate by the `nvcc -dlink` of
  # one of the executable.
  list(APPEND PUBLIC_DEPS VecGeom::vecgeom)
endif()

if(CELERITAS_USE_HIP)
  # Compile with HIP instead of CUDA (since we don't change our source
  # extensions)
  set_source_files_properties(
    ${DEVICE_SOURCES}
    PROPERTIES LANGUAGE HIP
  )
endif()

celeritas_add_library(celeritas ${SOURCES} ${DEVICE_SOURCES})
celeritas_strip_alias(_TARGET celeritas)
add_library(Celeritas::Core ALIAS ${_TARGET})
add_library(Celeritas::celeritas ALIAS ${_TARGET})

# Require at least C++14
set_property(TARGET ${_TARGET} PROPERTY CXX_STANDARD 14)

if(CELERITAS_USE_ROOT AND NOT BUILD_SHARED_LIBS)
  # Must build with -fPIC to link with ROOT
  set_target_properties(celeritas
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )
endif()

if(NOT CELERITAS_USE_OpenMP
    AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
         OR CMAKE_CXX_COMPILER_ID MATCHES "Clang$"))
  celeritas_target_compile_options(celeritas
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
  )
endif()

celeritas_target_link_libraries(celeritas
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

celeritas_target_include_directories(celeritas
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

celeritas_install(TARGETS celeritas
  EXPORT celeritas-targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  COMPONENT runtime
)

#----------------------------------------------------------------------------#
# GEANT/ROOT I/O MODULE
#----------------------------------------------------------------------------#

set(_IO_LIBTYPE) # empty for automatic
set(_IO_SOURCES)
set(_IO_PRIVATE_DEPS celeritas)

if(CELERITAS_USE_ROOT)
  include(${ROOT_USE_FILE})

  root_generate_dictionary(CeleritasRootInterface
    celeritas/io/ImportData.hh
    celeritas/io/ImportElement.hh
    celeritas/io/ImportMaterial.hh
    celeritas/io/ImportParticle.hh
    celeritas/io/ImportPhysicsTable.hh
    celeritas/io/ImportPhysicsVector.hh
    celeritas/io/ImportProcess.hh
    celeritas/io/ImportVolume.hh
    NOINSTALL
    MODULE celeritas_io
    LINKDEF celeritas/ext/RootInterfaceLinkDef.h
  )
  list(APPEND _IO_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/CeleritasRootInterface.cxx
    celeritas/ext/RootExporter.cc
    celeritas/ext/RootImporter.cc
    celeritas/ext/detail/TFileUniquePtr.root.cc
  )
  # Note that ROOT requires *shared* libraries due to runtime initialization,
  # *and* it must be forcibly linked into any ROOT-using application by ensuring
  # `-Wl,--no-as-needed` for executables (on Ubuntu, which defaults to linking
  # only as needed).
  set(_IO_LIBTYPE SHARED)
  list(APPEND _IO_PRIVATE_DEPS
    ROOT::Core ROOT::Tree
  )
endif()

if(CELERITAS_USE_Geant4)
  list(APPEND _IO_SOURCES
    celeritas/ext/GeantImporter.cc
    celeritas/ext/GeantSetup.cc
    celeritas/ext/detail/ActionInitialization.cc
    celeritas/ext/detail/DetectorConstruction.cc
    celeritas/ext/detail/GeantBremsstrahlungProcess.cc
    celeritas/ext/detail/GeantExceptionHandler.cc
    celeritas/ext/detail/GeantLoggerAdapter.cc
    celeritas/ext/detail/GeantPhysicsList.cc
    celeritas/ext/detail/ImportProcessConverter.cc
    celeritas/ext/detail/PrimaryGeneratorAction.cc
  )
  list(APPEND _IO_PRIVATE_DEPS
    XercesC::XercesC
    ${Geant4_LIBRARIES}
  )
endif()
if(CELERITAS_USE_JSON)
  list(APPEND _IO_SOURCES
    celeritas/ext/GeantSetupOptionsIO.json.cc
  )
endif()

if(NOT _IO_SOURCES)
  # No sources to compile: header only
  set(_IO_LIBTYPE INTERFACE)
  set(_IO_PRIVATE_DEPS)
endif()

add_library(celeritas_io ${_IO_LIBTYPE} ${_IO_SOURCES})
if(_IO_PRIVATE_DEPS)
  celeritas_target_link_libraries(celeritas_io
    PRIVATE ${_IO_PRIVATE_DEPS}
  )
endif()
add_library(Celeritas::IO ALIAS celeritas_io)
add_library(Celeritas::celeritas_io ALIAS celeritas_io)

celeritas_install(TARGETS celeritas_io
  EXPORT celeritas-targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  COMPONENT runtime
)

if(CELERITAS_USE_ROOT)
  # Install the rootmap/pcm files needed for users or downstream apps to use
  # Celeritas ROOT interfaces
  set(_iolib_prefix
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}celeritas_io"
  )
  install(FILES
    "${_iolib_prefix}.rootmap"
    "${_iolib_prefix}_rdict.pcm"
    COMPONENT runtime
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
endif()

#----------------------------------------------------------------------------#
# HEADERS
#----------------------------------------------------------------------------#

# C/generated headers
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/celeritas_config.h"
  "${CMAKE_CURRENT_BINARY_DIR}/celeritas_cmake_strings.h"
  COMPONENT development
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# C++ source headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT development
  FILES_MATCHING REGEX ".*\\.hh?$"
)

#---------------------------------------------------------------------------##
