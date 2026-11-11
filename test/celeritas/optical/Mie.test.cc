//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Mie.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <utility>

#include "corecel/io/Repr.hh"
#include "corecel/random/Histogram.hh"
#include "corecel/random/HistogramSampler.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MieData.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/ValidationUtils.hh"
#include "celeritas/optical/interactor/MieInteractor.hh"
#include "celeritas/optical/model/MieExecutor.hh"
#include "celeritas/optical/model/MieModel.hh"

#include "InteractorHostTestBase.hh"
#include "OpticalMockTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MieTest : public InteractorHostBase, public OpticalMockTestBase
{
  protected:
    using HostDataCRef = HostCRef<MieData>;

    void SetUp() override
    {
        model_ = std::make_shared<MieModel const>(
            ActionId{0},
            this->imported_data().optical_physics.bulk.mie,
            this->optical_material());
        data_ = model_->host_ref();
    }

    OptMatId material_id_{0};
    Real3 direction_{0, 0, 1};
    std::shared_ptr<MieModel const> model_;
    HostCRef<MieData> data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MieTest, mie_params)
{
    // Test the material properties of mie scattering parameters
    MieMaterialData mie_record = data_.mie_record[material_id_];
    EXPECT_SOFT_EQ(0.99, mie_record.forward_g);
    EXPECT_SOFT_EQ(0.99, mie_record.backward_g);
    EXPECT_SOFT_EQ(0.80, mie_record.forward_ratio);
}

TEST_F(MieTest, mie_basic)
{
    int const num_samples = 4;

    MieInteractor interact(
        data_, this->particle_track(), direction_, material_id_);

    auto& rng_engine = this->InteractorHostBase::rng();
    // auto expected_rng_engine_count = rng_engine.count();
    this->set_inc_polarization({0, 1, 0});

    std::vector<real_type> dir_angle;
    std::vector<real_type> pol_angle;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->check_direction_polarization(result);

        // Store dot products with incident direction/polarization
        dir_angle.push_back(dot_product(result.direction, this->direction()));
        pol_angle.push_back(dot_product(
            result.polarization, this->particle_track().polarization()));
    }

    if constexpr (std::is_same_v<real_type, double>)
    {
        static real_type const expected_dir_angle[] = {
            -0.997467127484242,
            0.999530487034177,
            -0.999999642467185,
            0.996187032055894,
        };
        static real_type const expected_pol_angle[] = {
            0.999904430863429,
            -0.99959742953257,
            -0.999999650643697,
            0.996510957439599,
        };

        EXPECT_VEC_SOFT_EQ(expected_dir_angle, dir_angle);
        EXPECT_VEC_SOFT_EQ(expected_pol_angle, pol_angle);
        EXPECT_EQ(32, rng_engine.count());
    }
}

TEST_F(MieTest, mfp)
{
    OwningGridAccessor storage;

    auto builder = storage.create_mfp_builder();
    for (auto mat : range(OptMatId(this->num_optical_materials())))
    {
        model_->build_mfps(mat, builder);
    }

    EXPECT_TABLE_EQ(this->get_mfp_table(ImportModelClass::mie),
                    storage(builder.grid_ids()));
}

TEST_F(MieTest, stress_test)
{
    constexpr size_type num_samples = 1'000'000;
    auto& rng_engine = this->InteractorHostBase::rng();

    // The scattering angle is relative to the incident direction, so the
    // distributions must not depend on which way the photon was travelling:
    // +z, where the frame rotation is the identity, and an oblique direction
    // with no zero component
    Real3 const oblique = make_unit_vector(Real3{1, 2, 3});
    Real3 const oblique_pol = make_unit_vector(Real3{2, -1, 0});
    for (auto const& [inc_dir, inc_pol] :
         {std::pair{Real3{0, 0, 1}, Real3{0, 1, 0}},
          std::pair{oblique, oblique_pol}})
    {
        SCOPED_TRACE(repr(inc_dir));
        this->set_inc_polarization(inc_pol);
        MieInteractor interact(
            data_, this->particle_track(), inc_dir, material_id_);

        Histogram accum_dir{8, {-1, 1}};
        Histogram accum_pol{8, {-1, 1}};
        // Rounding can put the cosine of an oblique pair just past 1
        auto cosine = [](Real3 const& a, Real3 const& b) {
            return std::clamp(dot_product(a, b), real_type{-1}, real_type{1});
        };
        accumulate_n(
            [&](auto&& result) {
                accum_dir(cosine(result.direction, inc_dir));
                accum_pol(cosine(result.polarization,
                                 this->particle_track().polarization()));
            },
            interact,
            rng_engine,
            num_samples);
        EXPECT_FALSE(accum_dir.underflow() || accum_dir.overflow()
                     || accum_pol.underflow() || accum_pol.overflow());

        static double const expected_accum_dir[] = {
            0.792728,
            0.001868,
            0.002324,
            0.002896,
            0.0042,
            0.00708,
            0.0164,
            3.172504,
        };
        static double const expected_accum_pol[] = {
            1.992904,
            0.004736,
            0.001632,
            0.003144,
            0.002992,
            0.001624,
            0.004784,
            1.993412,
        };

        auto avg_samples = static_cast<double>(rng_engine.exchange_count())
                           / static_cast<double>(num_samples);
        SoftEqual<double> tol{1e-2, 1e-2};
        EXPECT_VEC_NEAR(expected_accum_dir, accum_dir.calc_density(), tol);
        EXPECT_VEC_NEAR(expected_accum_pol, accum_pol.calc_density(), tol);
        EXPECT_SOFT_NEAR(
            4.0 * sizeof(real_type) / sizeof(float), avg_samples, tol);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
