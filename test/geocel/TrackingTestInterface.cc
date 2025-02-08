//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/TrackingTestInterface.cc
//---------------------------------------------------------------------------//
#include "TrackingTestInterface.hh"

#include "corecel/io/Repr.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void GenericGeoTrackingResult::print_expected()
{
    using std::cout;
    cout
        << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
           "static char const* const expected_volumes[] = "
        << repr(this->volumes)
        << ";\n"
           "EXPECT_VEC_EQ(expected_volumes, result.volumes);\n"
           "static char const* const expected_volume_instances[] = "
        << repr(this->volume_instances)
        << ";\n"
           "EXPECT_VEC_EQ(expected_volume_instances, "
           "result.volume_instances);\n"
           "static real_type const expected_distances[] = "
        << repr(this->distances)
        << ";\n"
           "EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);\n"
           "static real_type const expected_hw_safety[] = "
        << repr(this->halfway_safeties)
        << ";\n"
           "EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);\n"
           "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
