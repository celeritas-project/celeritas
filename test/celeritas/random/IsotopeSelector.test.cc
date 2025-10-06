//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/IsotopeSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/IsotopeSelector.hh"

#include <memory>
#include <random>

#include "corecel/random/SequenceEngine.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialTestBase.hh"

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

TEST_F(IsotopeSelectorTest, iodide)
{
    // Sodium iodide: Iodide has 3 isotopes (fractions 0.05, 0.15, and 0.8)
    MaterialView mat_nai(host_mats, mats->find_material("NaI"));
    auto const& el_view = mat_nai.element_record(ElementComponentId{1});
    EXPECT_EQ(AtomicNumber{53}, el_view.atomic_number());

    auto select_iso = make_isotope_selector(el_view);

    std::vector<size_type> count(el_view.num_isotopes());
    for ([[maybe_unused]] auto i : range(1000))
    {
        IsotopeComponentId iso_id = select_iso(rng);
        ASSERT_LT(iso_id, count.size());
        count[iso_id.get()]++;
    }

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static unsigned long const expected_count[] = {49ul, 149ul, 802ul};
        EXPECT_VEC_EQ(expected_count, count);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
