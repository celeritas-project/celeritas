//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/OpticalDistributionIO.test.cc
//---------------------------------------------------------------------------//
#include <vector>

#include "corecel/random/distribution/PoissonDistribution.hh"
#include "celeritas/io/OpticalDistributionReader.hh"
#include "celeritas/io/OpticalDistributionWriter.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class OpticalDistributionIOTest : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using GT = GeneratorType;
    using Speed = units::LightSpeed;
    using VecDistribution = OpticalDistributionReader::VecDistribution;
    //!@}

  protected:
    VecDistribution make_distributions(size_type count)
    {
        std::mt19937 rng;

        std::vector<GT> types{GT::cherenkov, GT::scintillation};
        PoissonDistribution<real_type> sample_num_photons(100);

        optical::GeneratorDistributionData data;
        data.step_length = 0.2;
        data.charge = units::ElementaryCharge{-1};
        data.material = OptMatId(0);
        data.continuous_edep_fraction = 1;
        data.points[StepPoint::pre] = {Speed(0.7), 0, Real3{0, 0, 0}};
        data.points[StepPoint::post] = {Speed(0.6), 1e-11, Real3{0, 0, 0.2}};

        VecDistribution result(count, data);
        for (auto i : range(count))
        {
            result[i].type = types[i % types.size()];
            result[i].num_photons = sample_num_photons(rng);
            CELER_ASSERT(result[i]);
        }
        return result;
    }
};

TEST_F(OpticalDistributionIOTest, write_read)
{
    std::string filename = this->make_unique_filename(".jsonl");
    auto distributions = this->make_distributions(32);

    OpticalDistributionWriter write(filename);
    write(distributions);

    OpticalDistributionReader read(filename);
    auto result = read();

    EXPECT_EQ(distributions.size(), result.size());
    for (size_type i = 0; i < distributions.size(); ++i)
    {
        auto const& d = distributions[i];
        auto const& r = result[i];
        EXPECT_EQ(d.type, r.type);
        EXPECT_EQ(d.num_photons, r.num_photons);
        EXPECT_EQ(d.primary, r.primary);
        EXPECT_EQ(d.step_length, r.step_length);
        EXPECT_EQ(d.charge, r.charge);
        EXPECT_EQ(d.material, r.material);
        EXPECT_EQ(d.continuous_edep_fraction, r.continuous_edep_fraction);

        for (auto sp : {StepPoint::pre, StepPoint::pre})
        {
            auto const& d_sp = d.points[sp];
            auto const& r_sp = r.points[sp];
            EXPECT_EQ(d_sp.speed, r_sp.speed);
            EXPECT_EQ(d_sp.time, r_sp.time);
            EXPECT_EQ(d_sp.pos, r_sp.pos);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
