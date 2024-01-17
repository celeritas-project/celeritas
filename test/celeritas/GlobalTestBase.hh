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

class ActionRegistry;
class AtomicRelaxationParams;
class CutoffParams;
class ExplicitActionInterface;
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;
class SimParams;
class TrackInitParams;

class CoreParams;
class OutputRegistry;

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

    using SPConstGeo = SP<GeoParams const>;
    using SPConstMaterial = SP<MaterialParams const>;
    using SPConstGeoMaterial = SP<GeoMaterialParams const>;
    using SPConstParticle = SP<ParticleParams const>;
    using SPConstCutoff = SP<CutoffParams const>;
    using SPConstPhysics = SP<PhysicsParams const>;
    using SPConstAction = SP<ExplicitActionInterface const>;
    using SPConstRng = SP<RngParams const>;
    using SPConstSim = SP<SimParams const>;
    using SPConstTrackInit = SP<TrackInitParams const>;
    using SPConstCore = SP<CoreParams const>;

    using SPActionRegistry = SP<ActionRegistry>;
    using SPOutputRegistry = SP<OutputRegistry>;
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
    inline SPActionRegistry const& action_reg();
    inline SPConstCore const& core();

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
    inline SPActionRegistry const& action_reg() const;
    inline SPConstCore const& core() const;
    //!@}

    //// OUTPUT ////

    //! Access output manager
    SPOutputRegistry const& output_reg() const { return output_reg_; }
    //! Write output to a debug text file
    void write_output();
    //! Write output to a stream
    void write_output(std::ostream& os) const;

  protected:
    [[nodiscard]] virtual SPConstGeo build_geometry() = 0;
    [[nodiscard]] virtual SPConstMaterial build_material() = 0;
    [[nodiscard]] virtual SPConstGeoMaterial build_geomaterial() = 0;
    [[nodiscard]] virtual SPConstParticle build_particle() = 0;
    [[nodiscard]] virtual SPConstCutoff build_cutoff() = 0;
    [[nodiscard]] virtual SPConstPhysics build_physics() = 0;
    [[nodiscard]] virtual SPConstSim build_sim() = 0;
    [[nodiscard]] virtual SPConstTrackInit build_init() = 0;
    [[nodiscard]] virtual SPConstAction build_along_step() = 0;

  private:
    SPConstRng build_rng() const;
    SPActionRegistry build_action_reg() const;
    SPConstCore build_core();

  private:
    SPConstGeo geometry_;
    SPConstMaterial material_;
    SPConstGeoMaterial geomaterial_;
    SPConstParticle particle_;
    SPConstCutoff cutoff_;
    SPConstPhysics physics_;
    SPActionRegistry action_reg_;
    SPConstAction along_step_;
    SPConstRng rng_;
    SPConstSim sim_;
    SPConstTrackInit init_;
    SPConstCore core_;
    SPOutputRegistry output_reg_;
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
DEF_GTB_ACCESSORS(SPConstCore, core)

#undef DEF_GTB_ACCESSORS

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
