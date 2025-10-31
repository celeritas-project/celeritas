//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "geocel/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

#include "Test.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
class ObjectInterface;

namespace detail
{
class CsgUnitBuilder;
struct CsgUnit;
}  // namespace detail

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Store a unit and its builder over the lifetime of the class.
 */
class ObjectTestBase : public ::celeritas::test::Test
{
  public:
    //!@{
    //! \name Type aliases
    using Unit = detail::CsgUnit;
    using UnitBuilder = detail::CsgUnitBuilder;
    using Tol = Tolerance<>;
    //!@}

  public:
    //! Construction tolerance used by unit builder
    virtual Tol tolerance() const = 0;

    // Access the constructed unit
    inline Unit const& unit() const;

    // Lazily create and access the unit builder for passing around
    inline UnitBuilder& unit_builder();

    // Access the unit builder for const passing
    inline UnitBuilder const& unit_builder() const;

    // Create a new unit and unit builder
    void reset();

    // Create a new unit and unit builder with a known maximum extent
    void reset(BBox const& extents);

    // Construct a volume
    LocalVolumeId build_volume(ObjectInterface const& s);

    // Print the unit we've constructed
    void print_expected() const;

    // Print for ORANGE CSG-to-dot
    void print_csg() const;

  private:
    std::shared_ptr<Unit> unit_;
    std::shared_ptr<UnitBuilder> builder_;
    std::vector<std::string> volume_names_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
////---------------------------------------------------------------------------//
/*!
 * Access the constructed unit.
 */
auto ObjectTestBase::unit() const -> Unit const&
{
    CELER_EXPECT(unit_);
    return *unit_;
}

////---------------------------------------------------------------------------//
/*!
 * Lazily create and access the unit builder for passing around.
 */
auto ObjectTestBase::unit_builder() -> UnitBuilder&
{
    if (!builder_)
    {
        this->reset();
    }
    CELER_ENSURE(builder_);
    return *builder_;
}

//---------------------------------------------------------------------------//
/*!
 * Access the unit builder for const passing.
 */
auto ObjectTestBase::unit_builder() const -> UnitBuilder const&
{
    CELER_ENSURE(builder_);
    return *builder_;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
