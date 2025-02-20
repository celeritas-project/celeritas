//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Problem.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace inp
{
struct Problem;
}

class CoreParams;
class GeantSd;
class OpticalCollector;
class RootFileManager;
class StepCollector;
struct ImportData;

namespace setup
{
//---------------------------------------------------------------------------//
//! Result from loaded standalone input to be used in front-end apps
struct ProblemLoaded
{
    //! Problem setup
    std::shared_ptr<CoreParams> core_params;

    //!@{
    //! \name Input-dependent products

    //! Step collector
    std::shared_ptr<StepCollector> step_collector;
    //! Optical offload management
    std::shared_ptr<OpticalCollector> optical_collector;
    //! Geant4 SD interface
    std::shared_ptr<GeantSd> geant_sd;
    //! ROOT file manager
    std::shared_ptr<RootFileManager> root_manager;

    //!@}

    //!@{
    //! \name Temporary: to be used downstream
    //! \todo These should be refactored: should be built in Problem

    //! Write offloaded primaries
    std::string offload_file;
    //! Write diagnostic output
    std::string output_file;

    //!@}
};

//---------------------------------------------------------------------------//
// Set up the problem
ProblemLoaded problem(inp::Problem const& p, ImportData const& imported);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
