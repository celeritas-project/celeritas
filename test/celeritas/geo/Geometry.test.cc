//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/Geometry.test.cc
//---------------------------------------------------------------------------//
#include "Geometry.test.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/Ref.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/random/RngParams.hh"

#include "../GlobalGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeometryTest : public celeritas_test::GlobalGeoTestBase
{
  public:
    //!@{
    //! \name Type aliases
    template<MemSpace M>
    using StateStore = CollectionStateStore<GeoTestStateData, M>;
    template<MemSpace M>
    using PathLengthRef
        = celeritas::Collection<real_type, Ownership::reference, M, VolumeId>;
    using SpanConstReal = Span<const real_type>;
    //!@}

  protected:
    SPConstParticle build_particle() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstCutoff   build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstPhysics  build_physics() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction   build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstMaterial build_material() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstGeoMaterial build_geomaterial() override
    {
        CELER_ASSERT_UNREACHABLE();
    }

    //// INTERFACE ////

    virtual GeoTestScalars build_scalars() const      = 0;
    virtual SpanConstReal  reference_avg_path() const = 0;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    GeoTestParamsData<Ownership::const_reference, M> build_params()
    {
        SPConstGeo geo = this->geometry();

        GeoTestParamsData<Ownership::const_reference, M> result;
        result.s                    = this->build_scalars();
        result.s.num_volumes        = geo->num_volumes();
        result.s.ignore_zero_safety = geo->supports_safety();
        CELER_ASSERT(result.s);

        result.geometry = get_ref<M>(*geo);
        result.rng      = get_ref<M>(*this->rng());
        return result;
    }

    template<MemSpace M>
    std::vector<real_type>
    get_avg_path(PathLengthRef<M> path, size_type num_states)
    {
        std::vector<real_type> result(path.size());
        Copier<real_type, M>   copy_to{path[AllItems<real_type, M>{}]};
        copy_to(MemSpace::host, make_span(result));

        real_type norm = 1 / real_type(num_states);
        for (real_type& r : result)
        {
            r *= norm;
        }
        return result;
    }
};

//---------------------------------------------------------------------------//

class TestEm3Test : public GeometryTest
{
  protected:
    const char* geometry_basename() const override { return "testem3-flat"; }

    GeoTestScalars build_scalars() const final
    {
        GeoTestScalars result;
        result.lower = {-19.77, -20, -20};
        result.upper = {19.43, 20, 20};
        return result;
    }

    SpanConstReal reference_avg_path() const final;
};

auto TestEm3Test::reference_avg_path() const -> SpanConstReal
{
    static const real_type orange_ref[] = {
        0,     17.5,  0.412, 0.553, 0.479, 0.651, 0.576, 0.478, 0.464, 0.452,
        0.571, 0.484, 0.601, 0.733, 0.609, 0.578, 0.578, 0.517, 0.633, 0.711,
        0.666, 0.771, 0.837, 0.882, 0.773, 1.018, 0.719, 0.734, 0.723, 0.720,
        0.706, 0.829, 0.847, 0.808, 0.830, 0.867, 0.803, 0.880, 0.750, 0.920,
        0.807, 0.842, 0.798, 0.736, 0.785, 0.752, 0.695, 0.680, 0.684, 0.672,
        0.737, 0.656, 0.748, 0.748, 0.672, 0.774, 0.855, 0.778, 0.790, 0.784,
        0.790, 0.785, 0.672, 0.658, 0.671, 0.633, 0.690, 0.810, 0.672, 0.728,
        0.693, 0.804, 0.743, 0.695, 0.782, 0.702, 0.680, 0.628, 0.662, 0.634,
        0.640, 0.580, 0.695, 0.617, 0.672, 0.633, 0.728, 0.707, 0.609, 0.575,
        0.527, 0.540, 0.492, 0.701, 0.512, 0.648, 0.571, 0.571, 0.456, 0.428,
        0.415, 0.284};

    return make_span(orange_ref);
}

//---------------------------------------------------------------------------//

class SimpleCmsTest : public GeometryTest
{
  protected:
    const char* geometry_basename() const override { return "simple-cms"; }

    GeoTestScalars build_scalars() const final
    {
        GeoTestScalars result;
        result.lower = {-30, -30, -700};
        result.upper = {30, 30, 700};
        return result;
    }

    SpanConstReal reference_avg_path() const final;
};

auto SimpleCmsTest::reference_avg_path() const -> SpanConstReal
{
    static const real_type orange_ref[]
        = {0, 32.4, 133., 59.0, 105., 81.2, 71.8, 40.3};

    return make_span(orange_ref);
}

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3Test, host)
{
    const size_type            num_states = 256;
    const size_type            num_steps  = 1024;
    auto                       params = this->build_params<MemSpace::host>();
    StateStore<MemSpace::host> state{params, num_states};

    GeoTestLauncher launch{params, state.ref()};
    for (auto tid : range(ThreadId{num_states}))
    {
        for (CELER_MAYBE_UNUSED auto step : range(num_steps))
        {
            launch(tid);
        }
    }

    auto avg_path = this->get_avg_path(state.ref().accum_path, num_states);
    // PRINT_EXPECTED(avg_path);
    EXPECT_VEC_NEAR(this->reference_avg_path(), avg_path, 0.01);
}

TEST_F(TestEm3Test, TEST_IF_CELER_DEVICE(device))
{
    const size_type num_states = 256 * 16;
    const size_type num_steps  = 1024;
    auto            params     = this->build_params<MemSpace::device>();
    StateStore<MemSpace::device> state{this->build_params<MemSpace::host>(),
                                       num_states};

    for (CELER_MAYBE_UNUSED auto step : range(num_steps))
    {
        g_test(params, state.ref());
    }

    auto avg_path = this->get_avg_path(state.ref().accum_path, num_states);
    PRINT_EXPECTED(avg_path);
}

//---------------------------------------------------------------------------//
// SIMPLECMS
//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, host)
{
    const size_type            num_states = 256;
    const size_type            num_steps  = 1024;
    auto                       params = this->build_params<MemSpace::host>();
    StateStore<MemSpace::host> state{params, num_states};

    GeoTestLauncher launch{params, state.ref()};
    for (auto tid : range(ThreadId{num_states}))
    {
        for (CELER_MAYBE_UNUSED auto step : range(num_steps))
        {
            launch(tid);
        }
    }

    auto avg_path = this->get_avg_path(state.ref().accum_path, num_states);
    // PRINT_EXPECTED(avg_path);
    EXPECT_VEC_NEAR(this->reference_avg_path(), avg_path, 0.01);
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
