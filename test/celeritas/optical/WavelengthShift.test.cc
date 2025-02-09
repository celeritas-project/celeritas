//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/WavelengthShift.test.cc
//---------------------------------------------------------------------------//
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/WavelengthShiftParams.hh"
#include "celeritas/optical/interactor/WavelengthShiftInteractor.hh"

#include "InteractorHostTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using TimeSecond = celeritas::RealQuantity<celeritas::units::Second>;

//---------------------------------------------------------------------------//
/*!
 * Tabulated spectrum as a function of the reemitted photon energy [MeV].
 */
Span<real_type const> get_energy()
{
    static real_type const energy[] = {1.65e-6, 2e-6, 2.4e-6, 2.8e-6, 3.26e-6};
    return make_span(energy);
}

Span<real_type const> get_spectrum()
{
    static real_type const spectrum[] = {0.15, 0.25, 0.50, 0.40, 0.02};
    return make_span(spectrum);
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class WavelengthShiftTest : public InteractorHostTestBase
{
  protected:
    using HostDataCRef = HostCRef<WavelengthShiftData>;

    void SetUp() override
    {
        // Build wavelength shift (WLS) property data from a test input
        ImportWavelengthShift wls;
        wls.mean_num_photons = 2;
        wls.time_constant = native_value_from(TimeSecond(1e-9));
        // Reemitted photon energy range (visible light)
        wls.component.x = {get_energy().begin(), get_energy().end()};
        // Reemitted photon energy spectrum
        wls.component.y = {get_spectrum().begin(), get_spectrum().end()};
        wls.component.vector_type = ImportPhysicsVectorType::free;

        WavelengthShiftParams::Input input;
        input.data.push_back(std::move(wls));
        params_ = std::make_shared<WavelengthShiftParams>(input);
        data_ = params_->host_ref();
    }

    OpticalMaterialId material_id_{0};
    std::shared_ptr<WavelengthShiftParams const> params_;
    HostDataCRef data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(WavelengthShiftTest, data)
{
    // Test the material properties of WLS
    WlsMaterialRecord wls_record = data_.wls_record[material_id_];
    EXPECT_SOFT_EQ(2, wls_record.mean_num_photons);
    EXPECT_SOFT_EQ(1 * units::nanosecond, wls_record.time_constant);

    // Test the vector property (emission spectrum) of WLS

    // Test the energy range and spectrum of emitted photons
    NonuniformGridCalculator calc_cdf(data_.energy_cdf[material_id_],
                                      data_.reals);
    auto const& energy = calc_cdf.grid();
    EXPECT_EQ(5, energy.size());
    EXPECT_SOFT_EQ(1.65e-6, energy.front());
    EXPECT_SOFT_EQ(3.26e-6, energy.back());

    auto calc_energy = calc_cdf.make_inverse();
    auto const& cdf = calc_energy.grid();
    EXPECT_SOFT_EQ(0, cdf.front());
    EXPECT_SOFT_EQ(1, cdf.back());

    EXPECT_SOFT_EQ(energy.front(), calc_energy(0));
    EXPECT_SOFT_EQ(energy.back(), calc_energy(1));

    std::vector<real_type> wls_energy;

    for ([[maybe_unused]] auto i : range(4))
    {
        wls_energy.push_back(calc_energy(generate_canonical(this->rng())));
    }

    static double const expected_wls_energy[] = {1.98638940166891e-06,
                                                 2.86983459900623e-06,
                                                 3.18637969114425e-06,
                                                 2.1060413486539e-06};

    EXPECT_VEC_SOFT_EQ(expected_wls_energy, wls_energy);
}

TEST_F(WavelengthShiftTest, wls_basic)
{
    int const num_samples = 4;
    this->resize_secondaries(num_samples * 2);
    auto& rng_engine = this->rng();

    // Interactor with an energy point within the input component range
    real_type test_energy = 2e-6;
    this->set_inc_energy(Energy{test_energy});

    WavelengthShiftInteractor interactor(data_,
                                         this->particle_track(),
                                         material_id_,
                                         this->secondary_allocator());

    std::vector<size_type> num_secondaries;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interactor(rng_engine);
        size_type num_emitted = result.secondaries.size();

        num_secondaries.push_back(num_emitted);

        EXPECT_EQ(Interaction::Action::absorbed, result.action);
        for (auto j : range(num_emitted))
        {
            EXPECT_LT(result.secondaries[j].energy.value(), test_energy);
            EXPECT_SOFT_EQ(0,
                           dot_product(result.secondaries[j].polarization,
                                       result.secondaries[j].direction));
        }
    }
    static size_type const expected_num_secondaries[] = {1, 4, 3, 0};
    EXPECT_VEC_EQ(expected_num_secondaries, num_secondaries);
}

TEST_F(WavelengthShiftTest, wls_stress)
{
    int const num_samples = 128;

    WlsMaterialRecord wls_record = data_.wls_record[material_id_];
    this->resize_secondaries(
        num_samples * static_cast<int>(wls_record.mean_num_photons) * 4);
    auto& rng_engine = this->rng();
    Real3 const inc_dir = {0, 0, 1};

    std::vector<real_type> avg_emitted;
    std::vector<real_type> avg_energy;
    std::vector<real_type> avg_costheta;
    std::vector<real_type> avg_orthogonality;
    std::vector<real_type> avg_time;

    // Interactor with points above the reemission spectrum
    for (real_type inc_e : {5., 10., 50., 100.})
    {
        this->set_inc_energy(Energy{inc_e});
        WavelengthShiftInteractor interactor(data_,
                                             this->particle_track(),
                                             material_id_,
                                             this->secondary_allocator());

        size_type sum_emitted{};
        real_type sum_energy{};
        real_type sum_costheta{};
        real_type sum_orthogonality{};
        real_type sum_time{};

        for ([[maybe_unused]] int i : range(num_samples))
        {
            Interaction result = interactor(rng_engine);
            size_type num_emitted = result.secondaries.size();

            sum_emitted += num_emitted;
            for (auto j : range(num_emitted))
            {
                sum_energy += result.secondaries[j].energy.value();
                sum_costheta
                    += dot_product(result.secondaries[j].direction, inc_dir);
                sum_orthogonality
                    += dot_product(result.secondaries[j].polarization,
                                   result.secondaries[j].direction);
                sum_time += result.secondaries[j].time;
            }
        }
        avg_emitted.push_back(static_cast<double>(sum_emitted) / num_samples);
        avg_energy.push_back(sum_energy / sum_emitted);
        avg_costheta.push_back(sum_costheta / sum_emitted);
        avg_orthogonality.push_back(sum_orthogonality / sum_emitted);
        avg_time.push_back(sum_time / sum_emitted / units::second);
    }

    static double const expected_avg_emitted[]
        = {1.96875, 1.890625, 2.0234375, 2.0703125};

    static double const expected_avg_energy[] = {2.44571770464513e-06,
                                                 2.47500490691662e-06,
                                                 2.4162395900554e-06,
                                                 2.46151256760185e-06};

    static double const expected_avg_costheta[] = {0.0157611129315312,
                                                   0.0325629374415683,
                                                   0.0082191738981211,
                                                   0.0128202506207409};

    static double const expected_avg_orthogonality[] = {0, 0, 0, 0};

    static double const expected_avg_time[] = {1.08250310854364e-09,
                                               1.01943566892086e-09,
                                               1.05383398761085e-09,
                                               9.64465413967612e-10};

    EXPECT_VEC_EQ(expected_avg_emitted, avg_emitted);
    EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
    EXPECT_VEC_SOFT_EQ(expected_avg_orthogonality, avg_orthogonality);
    EXPECT_VEC_SOFT_EQ(expected_avg_time, avg_time);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
