//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantOpticalMatHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <G4Material.hh>

#include "celeritas/Types.hh"

#include "GeantMaterialPropertyGetter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class providing access to an optical material and its properties.
 *
 * Analogous to \c GeantSurfacePhysicsHelper but for optical materials
 * identified by \c OptMatId .
 */
class GeantOpticalMatHelper
{
  public:
    // Construct with optical material ID and corresponding G4Material pointer
    inline GeantOpticalMatHelper(OptMatId opt_mat_id,
                                 G4Material const* material);

    //! True if this helper is fully constructed and valid
    explicit operator bool() const { return material_ != nullptr; }

    // Get optical material ID
    OptMatId opt_mat_id() const { return opt_mat_id_; }

    // Get G4Material
    inline G4Material const& material() const;

    // Access the property getter for this material
    GeantMaterialPropertyGetter const& property_getter() const
    {
        CELER_ASSERT(*this);
        return getter_;
    }

  private:
    OptMatId opt_mat_id_;
    G4Material const* material_;
    GeantMaterialPropertyGetter getter_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with optical material ID and G4Material pointer.
 *
 * The material pointer must not be null.
 */
GeantOpticalMatHelper::GeantOpticalMatHelper(OptMatId opt_mat_id,
                                             G4Material const* material)
    : opt_mat_id_{opt_mat_id}
    , material_{material}
    , getter_{material->GetMaterialPropertiesTable(),
              material->GetName().c_str()}
{
    CELER_EXPECT(opt_mat_id_);
    CELER_EXPECT(material_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the G4Material for this optical material.
 */
G4Material const& GeantOpticalMatHelper::material() const
{
    CELER_ASSERT(*this);
    return *material_;
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Write material name and optical id to a stream
inline std::ostream&
operator<<(std::ostream& os, GeantOpticalMatHelper const& helper)
{
    os << helper.property_getter()
       << " (optical id=" << helper.opt_mat_id().unchecked_get() << ")";
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
