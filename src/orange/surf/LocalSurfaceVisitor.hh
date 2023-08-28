//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surface/LocalSurfaceVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeData.hh"

#include "SurfaceTypeTraits.hh"
#include "detail/AllSurfaces.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply a functor to a type-deleted local surface.
 *
 * An instance of this class is like \c std::visit but accepting a
 * \c LocalSurfaceId rather than a \c std::variant .
 *
 * Example: \code
 LocalSurfaceVisitor visit_surface{params_};
 auto sense = visit_surface(
    [&pos](auto const& s) { return s.calc_sense(pos); },
    surface_id);
 \endcode
 */
class LocalSurfaceVisitor
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    //!@}

  public:
    // Construct from ORANGE params and simple unit record
    inline CELER_FUNCTION
    LocalSurfaceVisitor(ParamsRef const& params,
                        SurfacesRecord const& local_surfaces);

    // Construct from ORANGE params and simple unit ID
    inline CELER_FUNCTION
    LocalSurfaceVisitor(ParamsRef const& params, SimpleUnitId unit);

    // Apply the function to the surface specified by the given ID
    template<class F>
    inline CELER_FUNCTION decltype(auto)
    operator()(F&& typed_visitor, LocalSurfaceId t);

  private:
    //// TYPES ////

    template<class T>
    using Items = Collection<T, Ownership::const_reference, MemSpace::native>;

    //// DATA ////

    ParamsRef const& params_;
    SurfacesRecord const& surfaces_;

    //// HELPER FUNCTIONS ////

    // Construct a surface from a data offset
    template<class T>
    inline CELER_FUNCTION T make_surface(LocalSurfaceId data_offset) const;

    // TODO: change surfaces record to use ItemMap, move this to Collection.hh
    template<class T, class U>
    static inline CELER_FUNCTION T get_item(Items<T> const& items,
                                            ItemRange<T> const& range,
                                            ItemId<U> item);

    template<class T>
    static inline CELER_FUNCTION T const*
    get_ptr(Items<T> const& items, ItemId<T> item);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from ORANGE data (inside simple unit tracker).
 */
CELER_FORCEINLINE_FUNCTION
LocalSurfaceVisitor::LocalSurfaceVisitor(ParamsRef const& params,
                                         SurfacesRecord const& local_surfaces)
    : params_{params}, surfaces_{local_surfaces}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from ORANGE data with surfaces from a simple unit.
 */
CELER_FORCEINLINE_FUNCTION
LocalSurfaceVisitor::LocalSurfaceVisitor(ParamsRef const& params,
                                         SimpleUnitId unit)
    : LocalSurfaceVisitor{params, params.simple_units[unit].surfaces}
{
}

//---------------------------------------------------------------------------//
/*!
 * Apply the function to the surface specified by the given ID.
 */
template<class F>
CELER_FUNCTION decltype(auto)
LocalSurfaceVisitor::operator()(F&& func, LocalSurfaceId id)
{
    CELER_EXPECT(id < surfaces_.size());

    // Apply type-deleted functor based on type
    return visit_surface_type(
        [this, &func, id](auto s_traits) {
            // Call the user-provided action using the reconstructed surface
            using S = typename decltype(s_traits)::type;
            return func(this->make_surface<S>(id));
        },
        this->get_item(params_.surface_types, surfaces_.types, id));
}

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a surface of a given type using the data at a specific ID.
 */
template<class T>
CELER_FUNCTION T LocalSurfaceVisitor::make_surface(LocalSurfaceId id) const
{
    CELER_EXPECT(id < surfaces_.size());

    OpaqueId<real_type> offset
        = this->get_item(params_.real_ids, surfaces_.data_offsets, id);
    constexpr size_type size{T::StorageSpan::extent};
    CELER_ASSERT(offset + size <= params_.reals.size());

    real_type const* data = this->get_ptr(params_.reals, offset);
    return T{Span<real_type const, size>{data, size}};
}

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the item in the given range.
 */
template<class T, class U>
CELER_FUNCTION T LocalSurfaceVisitor::get_item(Items<T> const& items,
                                               ItemRange<T> const& range,
                                               ItemId<U> item)
{
    CELER_EXPECT(*range.end() <= items.size());
    CELER_EXPECT(item < range.size());

    T const* ptr = LocalSurfaceVisitor::get_ptr(items, *range.begin());
    return *(ptr + item.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the item in the given range.
 */
template<class T>
CELER_FUNCTION T const*
LocalSurfaceVisitor::get_ptr(Items<T> const& items, ItemId<T> item)
{
    CELER_EXPECT(item < items.size());

    T const* ptr = items[AllItems<T>{}].data();
    return ptr + item.unchecked_get();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
