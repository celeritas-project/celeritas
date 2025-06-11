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
#include "celeritas/optical/interactor/WavelengthShiftGenerator.hh"
#include "celeritas/optical/interactor/WavelengthShiftInteractor.hh"
#include "celeritas/optical/model/WavelengthShiftModel.hh"

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

class WavelengthShiftTest : public InteractorHostBase,
                            public OpticalMockTestBase
{
  protected:
    using HostDataCRef = HostCRef<WavelengthShiftData>;

    void SetUp() override
    {
        auto const& data = this->imported_data();
        WavelengthShiftModel::Input input;
        input.model = ImportModelClass::wls;
        for (auto const& mat : data.optical_materials)
        {
            input.data.push_back(mat.wls);
        }
        auto models
            = std::make_shared<ImportedModels const>(data.optical_models);
        model_ = std::make_shared<WavelengthShiftModel const>(
            ActionId{0}, models, input);
        data_ = model_->host_ref();
    }

    OptMatId material_id_{0};
    Real3 position_{1, 2, 3};
    std::shared_ptr<WavelengthShiftModel const> model_;
    HostCRef<WavelengthShiftData> data_;
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

    auto& rng = this->InteractorHostBase::rng();
    for ([[maybe_unused]] auto i : range(4))
    {
        wls_energy.push_back(calc_energy(generate_canonical(rng)));
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
    auto& rng = this->InteractorHostBase::rng();

    // Interactor with an energy point within the input component range
    real_type test_energy = 2e-6;
    this->set_inc_energy(Energy{test_energy});

    WavelengthShiftInteractor interact(data_,
                                       this->particle_track(),
                                       this->sim_track(),
                                       position_,
                                       material_id_);

    std::vector<size_type> num_photons;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng);
        EXPECT_EQ(Interaction::Action::absorbed, result.action);

        size_type num_emitted = result.distribution.num_photons;
        num_photons.push_back(num_emitted);

        for (size_type j = 0; j < num_emitted; ++j)
        {
            auto photon
                = WavelengthShiftGenerator(data_, result.distribution)(rng);
            EXPECT_LT(photon.energy.value(), test_energy);
            EXPECT_SOFT_EQ(0,
                           dot_product(photon.polarization, photon.direction));
        }
    }
    static size_type const expected_num_photons[] = {1, 4, 3, 0};
    EXPECT_VEC_EQ(expected_num_photons, num_photons);
}

TEST_F(WavelengthShiftTest, wls_stress)
{
    int const num_samples = 128;
    auto& rng = this->InteractorHostBase::rng();

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
        WavelengthShiftInteractor interact(data_,
                                           this->particle_track(),
                                           this->sim_track(),
                                           position_,
                                           material_id_);

        size_type sum_emitted{};
        real_type sum_energy{};
        real_type sum_costheta{};
        real_type sum_orthogonality{};
        real_type sum_time{};

        for ([[maybe_unused]] int i : range(num_samples))
        {
            Interaction result = interact(rng);
            size_type num_emitted = result.distribution.num_photons;
            sum_emitted += num_emitted;

            for (size_type j = 0; j < num_emitted; ++j)
            {
                auto photon = WavelengthShiftGenerator(
                    data_, result.distribution)(rng);
                sum_energy += photon.energy.value();
                sum_costheta += dot_product(photon.direction, inc_dir);
                sum_orthogonality
                    += dot_product(photon.polarization, photon.direction);
                sum_time += photon.time;
            }
        }
        avg_emitted.push_back(static_cast<double>(sum_emitted) / num_samples);
        avg_energy.push_back(sum_energy / sum_emitted);
        avg_costheta.push_back(sum_costheta / sum_emitted);
        avg_orthogonality.push_back(sum_orthogonality / sum_emitted);
        avg_time.push_back(sum_time / sum_emitted / units::nanosecond);
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

    static double const expected_avg_time[] = {
        1.0825031085436, 1.0194356689209, 1.0538339876109, 0.96446541396761};

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
