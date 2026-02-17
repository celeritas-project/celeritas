//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractorHostTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/data/StateDataStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffData.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Secondary.hh"

// Test helpers
#include "corecel/random/DiagnosticRngEngine.hh"

#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct ImportProcess;
class ParticleTrackView;
class MaterialTrackView;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for EM physics models.
 *
 * \todo Since this now uses Collection objects it's generally safe to use this
 * to test Models as well as device code -- think about renaming it.
 */
class InteractorHostBase
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using MevEnergy = units::MevEnergy;
    using Action = Interaction::Action;
    using SecondaryAllocator = StackAllocator<Secondary>;
    using constSpanSecondaries = Span<Secondary const>;
    //!@}

  public:
    //!@{
    //! Initialize and destroy
    InteractorHostBase();
    ~InteractorHostBase();
    //!@}

    // Helper to make dummy ImportProcess
    ImportProcess
    make_import_process(PDGNumber particle,
                        PDGNumber secondary,
                        ImportProcessClass ipc,
                        std::vector<ImportModelClass> models,
                        std::vector<Array<double, 2>> model_limits) const;

    //!@{
    //! Set and get material properties
    void set_material_params(MaterialParams::Input inp);
    std::shared_ptr<MaterialParams const> const& material_params() const
    {
        CELER_EXPECT(material_params_);
        return material_params_;
    }
    //!@}

    //!@{
    //! Set and get particle params
    void set_particle_params(ParticleParams::Input inp);
    std::shared_ptr<ParticleParams const> const& particle_params() const
    {
        CELER_EXPECT(particle_params_);
        return particle_params_;
    }
    //!@}

    //!@{
    //! Set and get cutoff params
    void set_cutoff_params(CutoffParams::Input inp);
    std::shared_ptr<CutoffParams const> const& cutoff_params() const
    {
        CELER_EXPECT(cutoff_params_);
        return cutoff_params_;
    }
    //!@}

    //!@{
    //! Set and get imported processes
    void set_imported_processes(std::vector<ImportProcess> inp);
    std::shared_ptr<ImportedProcesses const> const& imported_processes() const
    {
        CELER_EXPECT(imported_processes_);
        return imported_processes_;
    }
    //!@}

    //!@{
    //! Material properties
    void set_material(std::string const& name);
    MaterialTrackView& material_track()
    {
        CELER_EXPECT(mt_view_);
        return *mt_view_;
    }
    //!@}

    //!@{
    //! Incident particle properties and access
    void set_inc_particle(PDGNumber n, MevEnergy energy);
    void set_inc_direction(Real3 const& dir);
    Real3 const& direction() const { return inc_direction_; }
    ParticleTrackView const& particle_track() const
    {
        CELER_EXPECT(pt_view_);
        return *pt_view_;
    }
    //!@}

    //!@{
    //! Secondary stack storage and access
    void resize_secondaries(int count);
    SecondaryAllocator& secondary_allocator()
    {
        CELER_EXPECT(sa_view_);
        return *sa_view_;
    }
    //!@}

    //!@{
    //! Get random number generator with clean counter
    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }
    //!@}

    // Check for energy and momentum conservation
    void check_conservation(Interaction const& interaction) const;

    // Check for energy conservation
    void check_energy_conservation(Interaction const& interaction) const;

    // Check for momentum conservation
    void check_momentum_conservation(Interaction const& interaction) const;

  private:
    template<template<Ownership, MemSpace> class S>
    using StateStore = StateDataStore<S, MemSpace::host>;
    template<Ownership W, MemSpace M>
    using SecondaryStackData = StackAllocatorData<Secondary, W, M>;

    std::shared_ptr<MaterialParams const> material_params_;
    std::shared_ptr<ParticleParams const> particle_params_;
    std::shared_ptr<CutoffParams const> cutoff_params_;
    std::shared_ptr<ImportedProcesses const> imported_processes_;
    RandomEngine rng_;

    StateStore<MaterialStateData> ms_;
    StateStore<ParticleStateData> ps_;

    Real3 inc_direction_ = {0, 0, 1};
    StateStore<SecondaryStackData> secondaries_;

    // Views
    std::shared_ptr<MaterialTrackView> mt_view_;
    std::shared_ptr<ParticleTrackView> pt_view_;
    std::shared_ptr<SecondaryAllocator> sa_view_;
};

class InteractorHostTestBase : public InteractorHostBase, public Test
{
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
