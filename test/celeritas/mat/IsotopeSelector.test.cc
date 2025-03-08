//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/IsotopeSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/mat/IsotopeSelector.hh"

#include <memory>
#include <random>

#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/random/SequenceEngine.hh"

#include "MaterialTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using MaterialParamsRef = MaterialParams::HostRef;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class IsotopeSelectorTest : public MaterialTestBase, public Test
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = std::mt19937;
    //!@}

  protected:
    void SetUp() override
    {
        mats = build_material();
        host_mats = mats->host_ref();
    }

    SPConstMaterial mats;
    MaterialParamsRef host_mats;
    RandomEngine rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(IsotopeSelectorTest, single_isotope)
{
    MaterialView material(host_mats, mats->find_material("NaI"));
    auto const& element_na = material.element_record(ElementComponentId{0});
    IsotopeSelector select_iso(element_na);

    // Must always return the same isotope
    for ([[maybe_unused]] auto i : range(100))
    {
        EXPECT_EQ(0, select_iso(rng).get());
    }
}

TEST_F(IsotopeSelectorTest, multiple_isotopes)
{
    std::size_t const num_loops = 1000;

    // Diatomic hydrogen: two isotopes (fractions 0.9 and 0.1)
    MaterialView mat_h2(host_mats, MaterialId{2});
    auto const& element_h = mat_h2.element_record(ElementComponentId{0});
    IsotopeSelector select_iso_h(element_h);

    double sampled_frac_1h = 0;
    double sampled_frac_2h = 0;
    for ([[maybe_unused]] auto i : range(num_loops))
    {
        auto const id = select_iso_h(rng);
        id == IsotopeComponentId{0} ? sampled_frac_1h++ : sampled_frac_2h++;
    }
    sampled_frac_1h /= num_loops;
    sampled_frac_2h /= num_loops;

    std::vector<real_type> expected_frac_h;
    for (auto const& iso_record : element_h.isotopes())
    {
        expected_frac_h.push_back(iso_record.fraction);
    }

    EXPECT_EQ(2, expected_frac_h.size());
    EXPECT_SOFT_NEAR(expected_frac_h[0], sampled_frac_1h, std::sqrt(num_loops));
    EXPECT_SOFT_NEAR(expected_frac_h[1], sampled_frac_2h, std::sqrt(num_loops));

    // Sodium iodide: Iodide has 3 isotopes (fractions 0.05, 0.15, and 0.8)
    MaterialView mat_nai(host_mats, mats->find_material("NaI"));
    auto const& element_i = mat_nai.element_record(ElementComponentId{1});
    IsotopeSelector select_iso_i(element_i);

    double sampled_frac_125i = 0;
    double sampled_frac_126i = 0;
    double sampled_frac_127i = 0;
    for ([[maybe_unused]] auto i : range(num_loops))
    {
        auto const id = select_iso_h(rng);
        if (id == IsotopeComponentId{0})
            sampled_frac_125i++;
        if (id == IsotopeComponentId{1})
            sampled_frac_126i++;
        if (id == IsotopeComponentId{2})
            sampled_frac_127i++;
    }
    sampled_frac_125i /= num_loops;
    sampled_frac_126i /= num_loops;
    sampled_frac_127i /= num_loops;

    std::vector<real_type> expected_frac_i;
    for (auto const& iso_record : element_i.isotopes())
    {
        expected_frac_i.push_back(iso_record.fraction);
    }

    EXPECT_EQ(3, expected_frac_i.size());
    EXPECT_SOFT_NEAR(
        expected_frac_i[0], sampled_frac_125i, std::sqrt(num_loops));
    EXPECT_SOFT_NEAR(
        expected_frac_i[1], sampled_frac_126i, std::sqrt(num_loops));
    EXPECT_SOFT_NEAR(
        expected_frac_i[2], sampled_frac_127i, std::sqrt(num_loops));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
