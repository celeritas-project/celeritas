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
struct OpticalProblem;
struct Problem;
}  // namespace inp

namespace optical
{
class GeneratorBase;
class Transporter;
}  // namespace optical

class ActionSequence;
class CoreParams;
class GeantSd;
class OffloadWriter;
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
    //! Combined EM and optical offload management
    std::shared_ptr<OpticalCollector> optical_collector;
    //! Geant4 SD interface
    std::shared_ptr<GeantSd> geant_sd;
    //! ROOT file manager
    std::shared_ptr<RootFileManager> root_manager;
    //! Action sequence
    std::shared_ptr<ActionSequence> actions;
    //!@}

    //!@{
    //! \name Temporary: to be used downstream

    //! Write offloaded primaries
    std::shared_ptr<OffloadWriter> offload_writer;
    //!@}
};

//---------------------------------------------------------------------------//
//! Result from loaded optical standalone input to be used in front-end apps
struct OpticalProblemLoaded
{
    //! Optical-only problem setup
    std::shared_ptr<optical::Transporter> transporter;
    //! Optical photon generation action
    std::shared_ptr<optical::GeneratorBase> generator;
};

//---------------------------------------------------------------------------//
// Set up the problem
ProblemLoaded problem(inp::Problem const& p, ImportData const& imported);
// Set up the optical-only problem
OpticalProblemLoaded
problem(inp::OpticalProblem const& p, ImportData const& imported);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
