//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/random/RngParamsFwd.hh"

#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template<class P, template<MemSpace M> class S>
class StepActionInterface;

class ActionRegistry;
class AtomicRelaxationParams;
class CutoffParams;
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;
class SimParams;
class TrackInitParams;
class AuxParamsRegistry;
class WentzelOKVIParams;

class CoreParams;
template<MemSpace M>
class CoreState;
class OutputRegistry;

namespace optical
{
class CerenkovParams;
class MaterialParams;
class ScintillationParams;
}  // namespace optical

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Lazily construct core parameters, individually or together.
 *
 * \note Inherit from this class (or \c GlobalGeoTestBase) using \c
 * virtual \c public so that tests can create mixins (see e.g. \c
 * SimpleStepperTest).
 */
class GlobalTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using SP = std::shared_ptr<T>;
    using CoreStepActionInterface = StepActionInterface<CoreParams, CoreState>;

    using SPConstGeo = SP<GeoParams const>;
    using SPConstMaterial = SP<MaterialParams const>;
    using SPConstGeoMaterial = SP<GeoMaterialParams const>;
    using SPConstParticle = SP<ParticleParams const>;
    using SPConstCutoff = SP<CutoffParams const>;
    using SPConstPhysics = SP<PhysicsParams const>;
    using SPConstAction = SP<CoreStepActionInterface const>;
    using SPConstRng = SP<RngParams const>;
    using SPConstSim = SP<SimParams const>;
    using SPConstTrackInit = SP<TrackInitParams const>;
    using SPConstWentzelOKVI = SP<WentzelOKVIParams const>;
    using SPConstCore = SP<CoreParams const>;

    using SPActionRegistry = SP<ActionRegistry>;
    using SPOutputRegistry = SP<OutputRegistry>;
    using SPUserRegistry = SP<AuxParamsRegistry>;

    using SPConstCerenkov = SP<optical::CerenkovParams const>;
    using SPConstOpticalMaterial = SP<optical::MaterialParams const>;
    using SPConstScintillation = SP<optical::ScintillationParams const>;
    //!@}

  public:
    // Create output manager on construction
    GlobalTestBase();
    // Print output on failure if available
    virtual ~GlobalTestBase();

    //// ACCESSORS ////

    //!@{
    //! Access lazily constructed objects.
    inline SPConstGeo const& geometry();
    inline SPConstMaterial const& material();
    inline SPConstGeoMaterial const& geomaterial();
    inline SPConstParticle const& particle();
    inline SPConstCutoff const& cutoff();
    inline SPConstPhysics const& physics();
    inline SPConstAction const& along_step();
    inline SPConstRng const& rng();
    inline SPConstSim const& sim();
    inline SPConstTrackInit const& init();
    inline SPConstWentzelOKVI const& wentzel();
    inline SPActionRegistry const& action_reg();
    inline SPUserRegistry const& aux_reg();
    inline SPConstCore const& core();
    inline SPConstCerenkov const& cerenkov();
    inline SPConstOpticalMaterial const& optical_material();
    inline SPConstScintillation const& scintillation();

    inline SPConstGeo const& geometry() const;
    inline SPConstMaterial const& material() const;
    inline SPConstGeoMaterial const& geomaterial() const;
    inline SPConstParticle const& particle() const;
    inline SPConstCutoff const& cutoff() const;
    inline SPConstPhysics const& physics() const;
    inline SPConstAction const& along_step() const;
    inline SPConstRng const& rng() const;
    inline SPConstSim const& sim() const;
    inline SPConstTrackInit const& init() const;
    inline SPConstWentzelOKVI const& wentzel() const;
    inline SPActionRegistry const& action_reg() const;
    inline SPUserRegistry const& aux_reg() const;
    inline SPConstCore const& core() const;
    inline SPConstCerenkov const& cerenkov() const;
    inline SPConstOpticalMaterial const& optical_material() const;
    inline SPConstScintillation const& scintillation() const;
    //!@}

    //// OUTPUT ////

    //! Access output manager
    SPOutputRegistry const& output_reg() const { return output_reg_; }
    //! Write output to a debug text file
    void write_output();

  protected:
    [[nodiscard]] virtual SPConstGeo build_geometry() = 0;
    [[nodiscard]] virtual SPConstMaterial build_material() = 0;
    [[nodiscard]] virtual SPConstGeoMaterial build_geomaterial() = 0;
    [[nodiscard]] virtual SPConstParticle build_particle() = 0;
    [[nodiscard]] virtual SPConstCutoff build_cutoff() = 0;
    [[nodiscard]] virtual SPConstPhysics build_physics() = 0;
    [[nodiscard]] virtual SPConstSim build_sim() = 0;
    [[nodiscard]] virtual SPConstTrackInit build_init() = 0;
    [[nodiscard]] virtual SPConstWentzelOKVI build_wentzel() = 0;
    [[nodiscard]] virtual SPConstAction build_along_step() = 0;
    [[nodiscard]] virtual SPConstCerenkov build_cerenkov() = 0;
    [[nodiscard]] virtual SPConstOpticalMaterial build_optical_material() = 0;
    [[nodiscard]] virtual SPConstScintillation build_scintillation() = 0;

    // Do not insert StatusChecker
    void disable_status_checker();

  private:
    SPConstRng build_rng() const;
    SPActionRegistry build_action_reg() const;
    SPUserRegistry build_aux_reg() const;
    SPConstCore build_core();

  private:
    SPConstGeo geometry_;
    SPConstMaterial material_;
    SPConstGeoMaterial geomaterial_;
    SPConstParticle particle_;
    SPConstCutoff cutoff_;
    SPConstPhysics physics_;
    SPActionRegistry action_reg_;
    SPUserRegistry aux_reg_;
    SPConstAction along_step_;
    SPConstRng rng_;
    SPConstSim sim_;
    SPConstTrackInit init_;
    SPConstWentzelOKVI wentzel_;
    SPConstCore core_;
    SPOutputRegistry output_reg_;
    SPConstCerenkov cerenkov_;
    SPConstOpticalMaterial optical_material_;
    SPConstScintillation scintillation_;
    bool insert_status_checker_{true};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#define DEF_GTB_ACCESSORS(CLS, NAME)                \
    auto GlobalTestBase::NAME() -> CLS const&       \
    {                                               \
        if (!this->NAME##_)                         \
        {                                           \
            this->NAME##_ = this->build_##NAME();   \
            CELER_ASSERT(this->NAME##_);            \
        }                                           \
        return this->NAME##_;                       \
    }                                               \
    auto GlobalTestBase::NAME() const -> CLS const& \
    {                                               \
        CELER_ASSERT(this->NAME##_);                \
        return this->NAME##_;                       \
    }

DEF_GTB_ACCESSORS(SPConstGeo, geometry)
DEF_GTB_ACCESSORS(SPConstMaterial, material)
DEF_GTB_ACCESSORS(SPConstGeoMaterial, geomaterial)
DEF_GTB_ACCESSORS(SPConstParticle, particle)
DEF_GTB_ACCESSORS(SPConstCutoff, cutoff)
DEF_GTB_ACCESSORS(SPConstPhysics, physics)
DEF_GTB_ACCESSORS(SPConstAction, along_step)
DEF_GTB_ACCESSORS(SPConstRng, rng)
DEF_GTB_ACCESSORS(SPConstSim, sim)
DEF_GTB_ACCESSORS(SPConstTrackInit, init)
DEF_GTB_ACCESSORS(SPActionRegistry, action_reg)
DEF_GTB_ACCESSORS(SPUserRegistry, aux_reg)
DEF_GTB_ACCESSORS(SPConstCore, core)
DEF_GTB_ACCESSORS(SPConstCerenkov, cerenkov)
DEF_GTB_ACCESSORS(SPConstOpticalMaterial, optical_material)
DEF_GTB_ACCESSORS(SPConstScintillation, scintillation)
auto GlobalTestBase::wentzel() -> SPConstWentzelOKVI const&
{
    if (!this->wentzel_)
    {
        this->wentzel_ = this->build_wentzel();
    }
    return this->wentzel_;
}
auto GlobalTestBase::wentzel() const -> SPConstWentzelOKVI const&
{
    return this->wentzel_;
}

#undef DEF_GTB_ACCESSORS

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
