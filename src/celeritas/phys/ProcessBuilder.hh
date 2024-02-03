//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ProcessBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/io/ImportProcess.hh"

#include "AtomicNumber.hh"
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
//! Options used for constructing built-in Celeritas processes
struct ProcessBuilderOptions
{
    bool brem_combined{false};
    BremsModelSelection brems_selection{BremsModelSelection::all};
};

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas EM processes from imported data.
 *
 * This factory class has a hardcoded map that takes a \c ImportProcessClass
 * and constructs a built-in EM process (which will then build corresponding
 * models). This map can be overridden or extended by the \c user_build
 * constructor argument, which is a mapping of process class to user-supplied
 * factory functions.
 *
 * The function can return a null process pointer (in which case the caller
 * *must* ignore it) to indicate that a process should be deliberately omitted
 * from Celeritas. This can be used to (for example) skip very-high-energy
 * processes if Celeritas offloads only tracks below some energy cutoff. See \c
 * WarnAndIgnoreProcess below for a helper function for this purpose.
 *
 * \note Imported data may have multiple duplicate "process" entries, one
 * per particle type, because that's how Geant4 manages processes.
 */
class ProcessBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using IPC = ImportProcessClass;
    using Options = ProcessBuilderOptions;
    using SPProcess = std::shared_ptr<Process>;
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

    //! Input argument for user-provided process construction
    struct UserBuildInput
    {
        SPConstMaterial material;
        SPConstParticle particle;
        SPConstImported imported;
    };

    //!@{
    //! \name User builder type aliases
    using UserBuildFunction = std::function<SPProcess(UserBuildInput const&)>;
    using UserBuildMap = std::unordered_map<IPC, UserBuildFunction>;
    //!@}

  public:
    // Get an ordered set of all available processes
    static std::set<IPC>
    get_all_process_classes(std::vector<ImportProcess> const& processes);

    // Construct from imported and shared data and user construction
    ProcessBuilder(ImportData const& data,
                   SPConstParticle particle,
                   SPConstMaterial material,
                   UserBuildMap user_build,
                   Options options);

    // Construct without custom user builders
    ProcessBuilder(ImportData const& data,
                   SPConstParticle particle,
                   SPConstMaterial material,
                   Options options);

    // Default destructor
    ~ProcessBuilder();

    // Create a process from the data
    SPProcess operator()(IPC ipc);

  private:
    //// DATA ////

    UserBuildInput input_;
    UserBuildMap user_build_map_;
    std::function<ImportSBTable(AtomicNumber)> read_sb_;
    std::function<ImportLivermorePE(AtomicNumber)> read_livermore_;

    BremsModelSelection selection_;
    bool brem_combined_;
    bool enable_lpm_;
    bool use_integral_xs_;
    real_type coulomb_screening_factor_;

    //// HELPER FUNCTIONS ////

    SPConstMaterial const material() const { return input_.material; }
    SPConstParticle const particle() const { return input_.particle; }
    SPConstImported const imported() const { return input_.imported; }

    auto build_annihilation() -> SPProcess;
    auto build_compton() -> SPProcess;
    auto build_conversion() -> SPProcess;
    auto build_coulomb() -> SPProcess;
    auto build_ebrems() -> SPProcess;
    auto build_eioni() -> SPProcess;
    auto build_msc() -> SPProcess;
    auto build_photoelectric() -> SPProcess;
    auto build_rayleigh() -> SPProcess;
};

//---------------------------------------------------------------------------//
/*!
 * Warn about a missing process and deliberately skip it.
 *
 * Example:
 * \code
  ProcessBuilder::UserBuildMap ubm;
  ubm.emplace(ImportProcessClass::coulomb_scat,
              WarnAndIgnoreProcess{ImportProcessClass::coulomb_scat});
 * \endcode
 */
struct WarnAndIgnoreProcess
{
    //// TYPES ////
    using argument_type = ProcessBuilder::UserBuildInput const&;
    using result_type = ProcessBuilder::SPProcess;

    //// DATA ////

    ImportProcessClass process;

    //// METHODS ////

    result_type operator()(argument_type) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
