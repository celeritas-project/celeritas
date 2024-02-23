//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectTestBase.cc
//---------------------------------------------------------------------------//
#include "ObjectTestBase.hh"

#include "orange/orangeinp/ObjectInterface.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/orangeinp/detail/CsgUnitBuilder.hh"
#include "orange/orangeinp/detail/VolumeBuilder.hh"

#include "CsgTestUtils.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Create a new unit and unit builder.
 */
void ObjectTestBase::reset()
{
    unit_ = std::make_shared<Unit>();
    builder_ = std::make_shared<UnitBuilder>(unit_.get(), this->tolerance());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume.
 */
LocalVolumeId ObjectTestBase::build_volume(ObjectInterface const& s)
{
    detail::VolumeBuilder vb{&this->unit_builder()};
    auto final_node = s.build(vb);
    return builder_->insert_volume(final_node);
}

//---------------------------------------------------------------------------//
/*!
 * Print the unit we've constructed.
 */
void ObjectTestBase::print_expected() const
{
    CELER_EXPECT(unit_);
    ::celeritas::orangeinp::test::print_expected(*unit_);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
