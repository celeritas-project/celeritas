//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/VisitGeantVolumes.test.cc
//---------------------------------------------------------------------------//
#include "geocel/g4/VisitGeantVolumes.hh"

#include <string>
#include <vector>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/io/Label.hh"

#include "GeantGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct LogicalVisitor
{
    std::vector<std::string> names;
    void operator()(G4LogicalVolume const& lv)
    {
        names.push_back(Label::from_geant(lv.GetName()).name);
    }
};

class VisitGeantVolumesTest : public GeantGeoTestBase
{
  public:
    using SpanStringView = Span<std::string_view const>;

    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }
};

//---------------------------------------------------------------------------//
class FourLevelsTest : public VisitGeantVolumesTest
{
    std::string geometry_basename() const override { return "four-levels"; }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, logical)
{
    LogicalVisitor visit;
    visit_geant_volumes(visit, *this->geometry()->world()->GetLogicalVolume());

    static char const* const expected_names[]
        = {"World", "Envelope", "Shape1", "Shape2"};
    EXPECT_VEC_EQ(expected_names, visit.names);
}

//---------------------------------------------------------------------------//
class MultiLevelTest : public VisitGeantVolumesTest
{
    std::string geometry_basename() const override { return "multi-level"; }
};

//---------------------------------------------------------------------------//

TEST_F(MultiLevelTest, logical)
{
    LogicalVisitor visit;
    visit_geant_volumes(visit, *this->geometry()->world()->GetLogicalVolume());

    static char const* const expected_names[]
        = {"World", "Shape2", "Envelope", "Shape1"};
    EXPECT_VEC_EQ(expected_names, visit.names);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
