//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/LocalSurfaceVisitor.hh
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
    // Construct from ORANGE params and surfaces redord
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
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from ORANGE data and local surfaces.
 *
 * This is meant to be called from inside a simple unit tracker.
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

#if !defined(__DOXYGEN__) || __DOXYGEN__ > 0x010908
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
#endif

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a surface of a given type using the data at a specific ID.
 */
template<class T>
CELER_FUNCTION T LocalSurfaceVisitor::make_surface(LocalSurfaceId id) const
{
    using Reals = decltype(params_.reals);
    using ItemIdT = typename Reals::ItemIdT;
    using ItemRangeT = typename Reals::ItemRangeT;
    ItemIdT offset
        = this->get_item(params_.real_ids, surfaces_.data_offsets, id);
    constexpr size_type size{T::StorageSpan::extent};
    CELER_ASSERT(offset + size <= params_.reals.size());
    auto storage_span = params_.reals[ItemRangeT{offset, offset + size}];
    // Convert to a statically sized span using the first() member template
    // function, then construct a surface, for LdgSpan to work correctly.
    return T{storage_span.template first<size>()};
}

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the item offset using the given range in a given
 * collection.
 */
template<class T, class U>
CELER_FORCEINLINE_FUNCTION T LocalSurfaceVisitor::get_item(
    Items<T> const& items, ItemRange<T> const& offsets, ItemId<U> item)
{
    CELER_EXPECT(*offsets.end() <= items.size());
    CELER_EXPECT(item < offsets.size());
    return items[offsets][item.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
