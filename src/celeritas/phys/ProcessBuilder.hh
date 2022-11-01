//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ProcessBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ImportedProcesses;
class MaterialParams;
class ParticleParams;

struct ImportData;
struct ImportLivermorePE;
struct ImportSBTable;

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas EM processes from imported data.
 *
 * Note that imported data may have multiple duplicate "process" entries, one
 * per particle type.
 */
class ProcessBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using SPProcess       = std::shared_ptr<Process>;
    using SPConstParticle = std::shared_ptr<const ParticleParams>;
    using SPConstMaterial = std::shared_ptr<const MaterialParams>;
    //!@}

    struct Options
    {
        bool brem_combined{false};
    };

  public:
    // Construct from imported and shared data
    ProcessBuilder(const ImportData& data,
                   Options           options,
                   SPConstParticle   particle,
                   SPConstMaterial   material);

    // Default destructor
    ~ProcessBuilder();

    // Create a process from the data
    SPProcess operator()(ImportProcessClass ipc);

  private:
    //// DATA ////

    std::shared_ptr<const ParticleParams>          particle_;
    std::shared_ptr<const MaterialParams>          material_;
    std::shared_ptr<ImportedProcesses>             processes_;
    std::function<ImportSBTable(AtomicNumber)>     read_sb_;
    std::function<ImportLivermorePE(AtomicNumber)> read_livermore_;

    bool brem_combined_;
    bool enable_lpm_;
    bool use_integral_xs_;

    //// HELPER FUNCTIONS ////

    auto build_annihilation() -> SPProcess;
    auto build_compton() -> SPProcess;
    auto build_conversion() -> SPProcess;
    auto build_ebrems() -> SPProcess;
    auto build_eioni() -> SPProcess;
    auto build_msc() -> SPProcess;
    auto build_photoelectric() -> SPProcess;
    auto build_rayleigh() -> SPProcess;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
