//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectTestBase.cc
//---------------------------------------------------------------------------//
#include "ObjectTestBase.hh"

#include <nlohmann/json.hpp>

#include "orange/orangeinp/CsgTreeIO.json.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
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
    this->reset(BBox::from_infinite());
}

//---------------------------------------------------------------------------//
/*!
 * Create a new unit and unit builder with a known maximum extent.
 */
void ObjectTestBase::reset(BBox const& extents)
{
    unit_ = std::make_shared<Unit>();
    builder_ = std::make_shared<UnitBuilder>(
        unit_.get(), this->tolerance(), extents);
    volume_names_.clear();
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume.
 */
LocalVolumeId ObjectTestBase::build_volume(ObjectInterface const& s)
{
    detail::VolumeBuilder vb{&this->unit_builder()};
    auto final_node = s.build(vb);
    auto result = builder_->insert_volume(final_node);
    // TODO: also add the volume name, but this will require redoing tests
#if 0
    builder_->insert_md(final_node, {s.label()});
#endif
    volume_names_.emplace_back(s.label());
    return result;
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
/*!
 * Print an output like `.csg.json'
 */
void ObjectTestBase::print_csg() const
{
    CELER_EXPECT(unit_);
    // Copy so we can modify
    auto unit = *unit_;
    simplify(&unit.tree, CsgTree::false_node_id() + 1);

    auto j = nlohmann::json::array({unit});
    auto& vols = j[0]["volumes"];
    CELER_VALIDATE(vols.size() == volume_names_.size(),
                   << "size mismatch: CSG volumes=" << vols.size()
                   << ", labels=" << volume_names_.size());
    for (auto i : range(vols.size()))
    {
        vols[i]["label"] = volume_names_[i];
    }

    std::cout << j.dump() << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
