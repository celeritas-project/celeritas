//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <gtest/gtest.h>

#include "corecel/Assert.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/random/RngParamsFwd.hh"

#include "Test.hh"

namespace celeritas
{
class ActionManager;
class AtomicRelaxationParams;
class CutoffParams;
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;
class TrackInitParams;

class CoreParams;
class OutputManager;
} // namespace celeritas

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Lazily construct core parameters, individually or together.
 *
 * \note Inherit from this class using \c virtual \c public so that tests can
 * create mixins (see e.g. \c SimpleStepperTest).
 */
class GlobalTestBase : public celeritas_test::Test
{
  public:
    //!@{
    //! Type aliases
    template<class T>
    using SP = std::shared_ptr<T>;

    using SPConstGeo         = SP<const celeritas::GeoParams>;
    using SPConstMaterial    = SP<const celeritas::MaterialParams>;
    using SPConstGeoMaterial = SP<const celeritas::GeoMaterialParams>;
    using SPConstParticle    = SP<const celeritas::ParticleParams>;
    using SPConstCutoff      = SP<const celeritas::CutoffParams>;
    using SPConstPhysics     = SP<const celeritas::PhysicsParams>;
    using SPConstRng         = SP<const celeritas::RngParams>;
    using SPConstCore        = SP<const celeritas::CoreParams>;

    using SPActionManager = SP<celeritas::ActionManager>;
    using SPOutputManager = SP<celeritas::OutputManager>;
    //!@}

  public:
    // Create output manager on construction
    GlobalTestBase();
    // Default destructor
    virtual ~GlobalTestBase();

    //// ACCESSORS ////

    //!@{
    //! Access lazily constructed objects.
    inline SPConstGeo const&         geometry();
    inline SPConstMaterial const&    material();
    inline SPConstGeoMaterial const& geomaterial();
    inline SPConstParticle const&    particle();
    inline SPConstCutoff const&      cutoff();
    inline SPConstPhysics const&     physics();
    inline SPConstRng const&         rng();
    inline SPActionManager const&    action_mgr();
    inline SPConstCore const&        core();

    inline SPConstGeo const&         geometry() const;
    inline SPConstMaterial const&    material() const;
    inline SPConstGeoMaterial const& geomaterial() const;
    inline SPConstParticle const&    particle() const;
    inline SPConstCutoff const&      cutoff() const;
    inline SPConstPhysics const&     physics() const;
    inline SPConstRng const&         rng() const;
    inline SPActionManager const&    action_mgr() const;
    inline SPConstCore const&        core() const;
    //!@}

    //// OUTPUT ////

    //! Access output manager
    SPOutputManager const& output_mgr() const { return output_; }
    //! Write output to a debug text file
    void write_output();
    //! Write output to a stream
    void write_output(std::ostream& os) const;

  protected:
    virtual const char* geometry_basename() const = 0;

    virtual SPConstMaterial    build_material()    = 0;
    virtual SPConstGeoMaterial build_geomaterial() = 0;
    virtual SPConstParticle    build_particle()    = 0;
    virtual SPConstCutoff      build_cutoff()      = 0;
    virtual SPConstPhysics     build_physics()     = 0;

  private:
    SPConstGeo      build_geometry() const;
    SPConstRng      build_rng() const;
    SPActionManager build_action_mgr() const;
    SPConstCore     build_core();

    void register_geometry_output() {}
    void register_material_output() {}
    void register_geomaterial_output() {}
    void register_particle_output() {}
    void register_cutoff_output() {}
    void register_physics_output();
    void register_rng_output() {}
    void register_action_mgr_output();
    void register_core_output() {}

  private:
    SPConstGeo         geometry_;
    SPConstMaterial    material_;
    SPConstGeoMaterial geomaterial_;
    SPConstParticle    particle_;
    SPConstCutoff      cutoff_;
    SPConstPhysics     physics_;
    SPActionManager    action_mgr_;
    SPConstRng         rng_;
    SPConstCore        core_;
    SPOutputManager    output_;

    //// LAZY GEOMETRY CONSTRUCTION AND CLEANUP FOR VECGEOM ////

    struct LazyGeo
    {
        std::string basename{};
        SPConstGeo  geo{};
    };

    static LazyGeo& lazy_geo();

    class CleanupGeoEnvironment : public ::testing::Environment
    {
        void SetUp() override {}
        void TearDown() override;
    };
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#define DEF_GTB_ACCESSORS(CLS, NAME)              \
    auto GlobalTestBase::NAME()->CLS const&       \
    {                                             \
        if (!this->NAME##_)                       \
        {                                         \
            this->NAME##_ = this->build_##NAME(); \
            CELER_ASSERT(this->NAME##_);          \
            this->register_##NAME##_output();     \
        }                                         \
        return this->NAME##_;                     \
    }                                             \
    auto GlobalTestBase::NAME() const->CLS const& \
    {                                             \
        CELER_ASSERT(this->NAME##_);              \
        return this->NAME##_;                     \
    }

DEF_GTB_ACCESSORS(SPConstGeo, geometry)
DEF_GTB_ACCESSORS(SPConstMaterial, material)
DEF_GTB_ACCESSORS(SPConstGeoMaterial, geomaterial)
DEF_GTB_ACCESSORS(SPConstParticle, particle)
DEF_GTB_ACCESSORS(SPConstCutoff, cutoff)
DEF_GTB_ACCESSORS(SPConstPhysics, physics)
DEF_GTB_ACCESSORS(SPConstRng, rng)
DEF_GTB_ACCESSORS(SPActionManager, action_mgr)
DEF_GTB_ACCESSORS(SPConstCore, core)

#undef DEF_GTB_ACCESSORS

//---------------------------------------------------------------------------//
} // namespace celeritas_test
