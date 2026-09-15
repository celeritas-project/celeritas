//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/WavelengthShift.test.cc
//---------------------------------------------------------------------------//
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/StateDataStore.hh"
#include "corecel/random/Histogram.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/optical/gen/WavelengthShiftGenerator.hh"
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
    using HostStateStore
        = StateDataStore<WlsGeneratorStateData, MemSpace::host>;
    using DistId = ItemId<WlsDistributionData>;

    void SetUp() override {}

    void build_model(WlsDistribution time_profile)
    {
        auto const& data = this->imported_data().optical_physics.bulk.wls;
        inp::OpticalBulkWavelengthShift input;
        input.time_profile = time_profile;
        input.materials[material_id_] = data.materials.at(material_id_);
        model_ = std::make_shared<WavelengthShiftModel const>(
            ActionId{0},
            AuxId{0},
            input,
            this->optical_material(),
            GeneratorType::wls);
        data_ = model_->host_ref();
    }

    OptMatId material_id_{0};
    Real3 position_{1, 2, 3};
    std::shared_ptr<WavelengthShiftModel const> model_;
    HostCRef<WavelengthShiftData> data_;
    HostStateStore aux_data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(WavelengthShiftTest, data)
{
    this->build_model(WlsDistribution::exponential);

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

TEST_F(WavelengthShiftTest, time_profile)
{
    auto& rng = this->InteractorHostBase::rng();

    WlsDistributionData dist;
    dist.type = GeneratorType::wls;
    dist.num_photons = 1000;
    dist.energy = Energy{2e-6};
    dist.time = 5.0 * units::nanosecond;
    dist.material = material_id_;

    {
        // Test delta time profile
        this->build_model(WlsDistribution::delta);

        real_type const expected_time
            = dist.time + data_.wls_record[dist.material].time_constant;
        for (size_type i = 0; i < dist.num_photons; ++i)
        {
            auto photon = WavelengthShiftGenerator(data_, dist)(rng);
            EXPECT_EQ(expected_time, photon.time);
        }
    }
    {
        // Test exponential time profile
        this->build_model(WlsDistribution::exponential);

        real_type time_ns = dist.time / units::nanosecond;
        Histogram bin(8, {time_ns, time_ns + 4});
        for (size_type i = 0; i < dist.num_photons; ++i)
        {
            auto photon = WavelengthShiftGenerator(data_, dist)(rng);
            bin(photon.time / units::nanosecond);
        }
        static double const expected_density[] = {
            0.80898876404494,
            0.49642492339122,
            0.25740551583248,
            0.17977528089888,
            0.10827374872319,
            0.083758937691522,
            0.036772216547497,
            0.028600612870276,
        };
        EXPECT_VEC_SOFT_EQ(expected_density, bin.calc_density());
        EXPECT_FALSE(bin.underflow());
    }
}

TEST_F(WavelengthShiftTest, wls_basic)
{
    this->build_model(WlsDistribution::exponential);

    int const num_samples = 4;
    aux_data_ = HostStateStore{StreamId(0), num_samples};

    auto& rng = this->InteractorHostBase::rng();

    // Interactor with an energy point within the input component range
    real_type test_energy = 2e-6;
    this->set_inc_energy(Energy{test_energy});

    std::vector<size_type> num_photons;

    for (auto dist_id : range(id_cast<DistId>(num_samples)))
    {
        WavelengthShiftInteractor interact(data_,
                                           aux_data_.ref(),
                                           this->particle_track(),
                                           this->sim_track(),
                                           position_,
                                           material_id_,
                                           dist_id);
        Interaction result = interact(rng);
        EXPECT_EQ(Interaction::Action::absorbed, result.action);

        auto const& distribution = aux_data_.ref().distributions[dist_id];
        size_type num_emitted = distribution.num_photons;
        num_photons.push_back(num_emitted);

        for (size_type j = 0; j < num_emitted; ++j)
        {
            auto photon = WavelengthShiftGenerator(data_, distribution)(rng);
            EXPECT_LT(photon.energy.value(), test_energy);
            EXPECT_SOFT_EQ(0,
                           dot_product(photon.polarization, photon.direction));
        }
    }
    static size_type const expected_num_photons[] = {1, 3, 2, 0};
    EXPECT_VEC_EQ(expected_num_photons, num_photons);
}

TEST_F(WavelengthShiftTest, wls_stress)
{
    this->build_model(WlsDistribution::exponential);

    int const num_samples = 128;
    aux_data_ = HostStateStore{StreamId(0), num_samples};

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

        size_type sum_emitted{};
        real_type sum_energy{};
        real_type sum_costheta{};
        real_type sum_orthogonality{};
        real_type sum_time{};

        for (auto dist_id : range(id_cast<DistId>(num_samples)))
        {
            auto& distribution = aux_data_.ref().distributions[dist_id];
            distribution = {};

            WavelengthShiftInteractor interact(data_,
                                               aux_data_.ref(),
                                               this->particle_track(),
                                               this->sim_track(),
                                               position_,
                                               material_id_,
                                               dist_id);
            Interaction result = interact(rng);
            EXPECT_EQ(Interaction::Action::absorbed, result.action);

            size_type num_emitted = distribution.num_photons;
            sum_emitted += num_emitted;

            for (size_type j = 0; j < num_emitted; ++j)
            {
                auto photon
                    = WavelengthShiftGenerator(data_, distribution)(rng);
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

    static double const expected_avg_emitted[] = {
        1.9296875,
        2.0390625,
        1.9921875,
        1.90625,
    };
    static double const expected_avg_energy[] = {
        2.4413387759069e-06,
        2.4904156826514e-06,
        2.4533746821139e-06,
        2.4613199385829e-06,
    };
    static double const expected_avg_costheta[] = {
        0.002757227538625,
        0.02641480631072,
        -0.0080660917048565,
        -0.091755149736537,
    };
    static double const expected_avg_orthogonality[] = {
        1.4003603538535e-17,
        -1.7103667600855e-17,
        -1.3355906544124e-17,
        -1.5707211674352e-17,
    };
    static double const expected_avg_time[] = {
        0.97328123052851,
        1.0942253717352,
        1.0827289356891,
        0.93884591640346,
    };
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
