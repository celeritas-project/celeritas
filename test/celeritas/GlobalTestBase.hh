//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/random/params/RngParamsFwd.hh"
#include "geocel/LazyGeantGeoManager.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/ActionInterface.hh"

#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

class AtomicRelaxationParams;
class CherenkovParams;
class CutoffParams;
class ExtendFromPrimariesAction;
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;
class ScintillationParams;
class SimParams;
class SurfaceParams;
class TrackInitParams;
class VolumeParams;
class WentzelOKVIParams;

class ActionRegistry;
class AuxParamsRegistry;
class OutputRegistry;

class CoreParams;
template<MemSpace M>
class CoreState;
class CoreStateInterface;

struct Primary;

namespace optical
{
class CoreParams;
class MaterialParams;
class PhysicsParams;
}  // namespace optical

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Lazily construct core parameters, individually or together.
 *
 * \note Inherit from this class using \c virtual \c public so that tests can
 * create mixins (see e.g. \c SimpleStepperTest).
 *
 * \todo Replace the construction with modifiers to \c celeritas::inp data
 * structures, and build the core geometry with \c celeritas::setup.
 */
class GlobalTestBase : public Test, public LazyGeantGeoManager
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using SP = std::shared_ptr<T>;

    using SPConstAction = SP<CoreStepActionInterface const>;
    using SPConstCoreGeo = SP<CoreGeoParams const>;
    using SPConstCutoff = SP<CutoffParams const>;
    using SPConstGeoMaterial = SP<GeoMaterialParams const>;
    using SPConstMaterial = SP<MaterialParams const>;
    using SPConstParticle = SP<ParticleParams const>;
    using SPConstPhysics = SP<PhysicsParams const>;
    using SPConstRng = SP<RngParams const>;
    using SPConstSim = SP<SimParams const>;
    using SPConstTrackInit = SP<TrackInitParams const>;
    using SPConstSurface = SP<SurfaceParams const>;
    using SPConstVolume = SP<VolumeParams const>;
    using SPConstWentzelOKVI = SP<WentzelOKVIParams const>;

    using SPActionRegistry = SP<ActionRegistry>;
    using SPOutputRegistry = SP<OutputRegistry>;
    using SPUserRegistry = SP<AuxParamsRegistry>;

    using SPConstCore = SP<CoreParams const>;

    using SPConstCherenkov = SP<CherenkovParams const>;
    using SPConstOpticalMaterial = SP<optical::MaterialParams const>;
    using SPOpticalParams = SP<optical::CoreParams>;
    using SPConstOpticalPhysics = SP<optical::PhysicsParams const>;
    using SPConstScintillation = SP<ScintillationParams const>;

    using SPConstPrimariesAction = SP<ExtendFromPrimariesAction const>;
    using SpanConstPrimary = Span<Primary const>;
    //!@}

  public:
    // Create output manager on construction
    GlobalTestBase();
    // Print output on failure if available
    virtual ~GlobalTestBase();

    //// ACCESSORS ////

    //!@{
    //! Access lazily constructed objects.
    inline SPConstCoreGeo const& geometry();
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
    inline SPConstCherenkov const& cherenkov();
    inline SPActionRegistry const& optical_action_reg();
    inline SPConstOpticalMaterial const& optical_material();
    inline SPOpticalParams const& optical_params();
    inline SPConstOpticalPhysics const& optical_physics();
    inline SPConstScintillation const& scintillation();

    inline SPConstCoreGeo const& geometry() const;
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
    inline SPConstCherenkov const& cherenkov() const;
    inline SPActionRegistry const& optical_action_reg() const;
    inline SPConstOpticalMaterial const& optical_material() const;
    inline SPOpticalParams const& optical_params() const;
    inline SPConstOpticalPhysics const& optical_physics() const;
    inline SPConstScintillation const& scintillation() const;
    //!@}

    SPConstPrimariesAction const& primaries_action();
    void
    insert_primaries(CoreStateInterface& state, SpanConstPrimary primaries);

    //// OUTPUT ////

    //! Access output manager
    SPOutputRegistry const& output_reg() const { return output_reg_; }
    //! Write output to a debug text file
    void write_output();

  protected:
    // GDML basename must be supplied
    using LazyGeantGeoManager::gdml_basename;

    [[nodiscard]] virtual SPConstMaterial build_material() = 0;
    [[nodiscard]] virtual SPConstCoreGeo build_geometry();
    [[nodiscard]] virtual SPConstGeoMaterial build_geomaterial() = 0;
    [[nodiscard]] virtual SPConstParticle build_particle() = 0;
    [[nodiscard]] virtual SPConstCutoff build_cutoff() = 0;
    [[nodiscard]] virtual SPConstPhysics build_physics() = 0;
    [[nodiscard]] virtual SPConstSim build_sim() = 0;
    [[nodiscard]] virtual SPConstTrackInit build_init() = 0;
    [[nodiscard]] virtual SPConstWentzelOKVI build_wentzel() = 0;
    [[nodiscard]] virtual SPConstAction build_along_step() = 0;
    [[nodiscard]] virtual SPConstCherenkov build_cherenkov() = 0;
    [[nodiscard]] virtual SPConstOpticalMaterial build_optical_material() = 0;
    [[nodiscard]] virtual SPConstOpticalPhysics build_optical_physics() = 0;
    [[nodiscard]] virtual SPConstScintillation build_scintillation() = 0;

    // Do not insert StatusChecker
    void disable_status_checker();

    // Access surface and volume; called during build_core
    SPConstSurface const& surface() const { return surface_; }
    SPConstVolume const& volume() const { return volume_; }

    // Implement LazyGeantGeoManager
    SPConstGeoI build_geo_from_geant(SPConstGeantGeo const&) const final;

    // Implement LazyGeantGeoManager, allowed when ORANGE without Geant4
    SPConstGeoI build_geo_from_gdml(std::string const& filename) const final;

  private:
    SPConstRng build_rng() const;
    SPActionRegistry build_action_reg() const;
    SPUserRegistry build_aux_reg() const;
    SPConstCore build_core();
    SPActionRegistry build_optical_action_reg() const;
    SPOpticalParams build_optical_params();

  private:
    SPConstCoreGeo geometry_;
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

    // NOTE: these may not be built
    SPConstSurface surface_;
    SPConstVolume volume_;

    SPConstCherenkov cherenkov_;
    SPActionRegistry optical_action_reg_;
    SPConstOpticalMaterial optical_material_;
    SPOpticalParams optical_params_;
    SPConstOpticalPhysics optical_physics_;
    SPConstScintillation scintillation_;

    SPConstPrimariesAction primaries_action_;
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

DEF_GTB_ACCESSORS(SPConstCoreGeo, geometry)
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
DEF_GTB_ACCESSORS(SPConstCherenkov, cherenkov)
DEF_GTB_ACCESSORS(SPActionRegistry, optical_action_reg)
DEF_GTB_ACCESSORS(SPConstOpticalMaterial, optical_material)
DEF_GTB_ACCESSORS(SPOpticalParams, optical_params)
DEF_GTB_ACCESSORS(SPConstOpticalPhysics, optical_physics)
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
