//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/IsotopeSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/mat/IsotopeSelector.hh"

#include <memory>
#include <random>

#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/random/SequenceEngine.hh"

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

class IsotopeSelectorTest : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = std::mt19937;
    //!@}

  protected:
    void SetUp() override
    {
        using units::AmuMass;
        using units::MevMass;

        MaterialParams::Input inp;

        // Using nuclear masses provided by Geant4 11.0.3
        inp.isotopes = {
            // H
            {AtomicNumber{1}, AtomicNumber{1}, MevMass{938.272}, "1H"},
            {AtomicNumber{1}, AtomicNumber{2}, MevMass{1875.61}, "2H"},
            // Na
            {AtomicNumber{11}, AtomicNumber{23}, MevMass{21409.2}, "23Na"},
            // I
            {AtomicNumber{53}, AtomicNumber{125}, MevMass{116321}, "125I"},
            {AtomicNumber{53}, AtomicNumber{126}, MevMass{117253}, "126I"},
            {AtomicNumber{53}, AtomicNumber{127}, MevMass{118184}, "127I"}};

        inp.elements = {
            // H
            {AtomicNumber{1},
             AmuMass{1.008},
             {{IsotopeId{0}, 0.5}, {IsotopeId{1}, 0.5}},
             "H"},
            // Na
            {AtomicNumber{11}, AmuMass{22.98976928}, {{IsotopeId{2}, 1}}, "Na"},
            // I
            {AtomicNumber{53},
             AmuMass{126.90447},
             {{IsotopeId{3}, 0.05}, {IsotopeId{4}, 0.15}, {IsotopeId{5}, 0.8}},
             "I"},
        };

        inp.materials = {// Sodium iodide
                         {2.948915064677e+22,
                          293.0,
                          MatterState::solid,
                          {{ElementId{1}, 0.5}, {ElementId{2}, 0.5}},
                          "NaI"},
                         // Diatomic hydrogen
                         {1.0739484359044669e+20,
                          100.0,
                          MatterState::gas,
                          {{ElementId{0}, 1.0}},
                          "H2"}};
        mats = std::make_shared<MaterialParams>(std::move(inp));
        host_mats = mats->host_ref();

        // Allocate storage
        storage.assign(mats->max_element_components(), -1);
    }

    std::shared_ptr<MaterialParams> mats;
    MaterialParamsRef host_mats;
    RandomEngine rng;
    std::vector<real_type> storage;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(IsotopeSelectorTest, single_isotope)
{
    MaterialView material(host_mats, mats->find_material("NaI"));
    auto const& element_na = material.make_element_view(ElementComponentId{0});
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
    MaterialView mat_h2(host_mats, mats->find_material("H2"));
    auto const& element_h = mat_h2.make_element_view(ElementComponentId{0});
    IsotopeSelector select_iso_h(element_h);

    double avg_iso_1h = 0;
    double avg_iso_2h = 0;
    for ([[maybe_unused]] auto i : range(num_loops))
    {
        auto const id = select_iso_h(rng);
        id == IsotopeComponentId{0} ? avg_iso_1h++ : avg_iso_2h++;
    }
    avg_iso_1h /= num_loops;
    avg_iso_2h /= num_loops;

    std::vector<real_type> expected_frac_h;
    for (auto const& iso_record : element_h.isotopes())
    {
        expected_frac_h.push_back(iso_record.fraction);
    }

    EXPECT_EQ(2, expected_frac_h.size());
    EXPECT_SOFT_NEAR(expected_frac_h[0], avg_iso_1h, std::sqrt(num_loops));
    EXPECT_SOFT_NEAR(expected_frac_h[1], avg_iso_2h, std::sqrt(num_loops));

    // Sodium iodide: Iodide has 3 isotopes (fractions 0.05, 0.15, and 081)
    MaterialView mat_nai(host_mats, mats->find_material("NaI"));
    auto const& element_i = mat_nai.make_element_view(ElementComponentId{1});
    IsotopeSelector select_iso_i(element_i);

    double avg_iso_125i = 0;
    double avg_iso_126i = 0;
    double avg_iso_127i = 0;
    for ([[maybe_unused]] auto i : range(num_loops))
    {
        auto const id = select_iso_h(rng);
        if (id == IsotopeComponentId{0})
            avg_iso_125i++;
        if (id == IsotopeComponentId{1})
            avg_iso_126i++;
        if (id == IsotopeComponentId{2})
            avg_iso_127i++;
    }
    avg_iso_125i /= num_loops;
    avg_iso_126i /= num_loops;
    avg_iso_127i /= num_loops;

    std::vector<real_type> expected_frac_i;
    for (auto const& iso_record : element_i.isotopes())
    {
        expected_frac_i.push_back(iso_record.fraction);
    }

    EXPECT_EQ(3, expected_frac_i.size());
    EXPECT_SOFT_NEAR(expected_frac_i[0], avg_iso_125i, std::sqrt(num_loops));
    EXPECT_SOFT_NEAR(expected_frac_i[1], avg_iso_126i, std::sqrt(num_loops));
    EXPECT_SOFT_NEAR(expected_frac_i[2], avg_iso_127i, std::sqrt(num_loops));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
