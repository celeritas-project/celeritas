//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_set>
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Traits class to access children and associated logical volume
template<class PV>
struct VolumeVisitorTraits
{
    using LV = void;

    static void get_children(PV const& parent, std::vector<PV const*>& dst);
    static LV get_lv(PV const& pv);
};

//---------------------------------------------------------------------------//
/*!
 * Recursively visit volumes.
 * \tparam T Volume type
 *
 * This class can be used for both Geant4 and VecGeom to give the same visiting
 * behavior across the two.
 *
 * The function must have the signature
 * <code>bool(*)(T const&, int)</code>
 * where the return value indicates whether the volume's children should be
 * visited, and the integer is the depth of the volume being visited.
 *
 * By default this will visit the entire "touchable" hierarchy: this may be
 * very expensive! If it's desired to only visit single physical volumes, mark
 * them as visited using a set (see unit test for example).
 */
template<class T>
class VolumeVisitor
{
  public:
    //! Construct from top-level volume
    explicit VolumeVisitor(T const& world) : world_{world} {}

    // Apply this visitor
    template<class F>
    inline void operator()(F&& visit);

  private:
    T const& world_;

    struct QueuedVolume
    {
        T const* pv{nullptr};
        int depth{0};
    };
};

//---------------------------------------------------------------------------//

// Visit all logical volumes
template<class F, class T>
inline void visit_logical_volumes(F&& vis, T const& parent_vol);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply this visitor.
 */
template<class T>
template<class F>
void VolumeVisitor<T>::operator()(F&& visit)
{
    using TraitsT = VolumeVisitorTraits<T>;

    std::vector<QueuedVolume> queue;
    std::vector<T const*> temp_children;
    auto visit_impl = [&](T const& pv, int depth) {
        if (visit(pv, depth))
        {
            temp_children.clear();
            TraitsT::get_children(pv, temp_children);

            // Append children in reverse order since we pop back
            for (auto iter = temp_children.rbegin();
                 iter != temp_children.rend();
                 ++iter)
            {
                queue.push_back({*iter, depth + 1});
            }
        }
    };

    // Visit the top-level physical volume
    visit_impl(world_, 0);

    while (!queue.empty())
    {
        QueuedVolume qv = queue.back();
        queue.pop_back();

        // Visit popped daughter
        visit_impl(*qv.pv, qv.depth);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Visit all logical volumes, once, depth-first.
 */
template<class F, class T>
inline void visit_logical_volumes(F&& vis, T const& parent_vol)
{
    using TraitsT = VolumeVisitorTraits<T>;
    using LV = typename TraitsT::LV;

    VolumeVisitor visit_impl{parent_vol};

    std::unordered_set<LV const*> visited;
    visit_impl([&vis, &visited](T const& pv, int) -> bool {
        auto const& lv = TraitsT::get_lv(pv);
        if (!visited.insert(&lv).second)
        {
            // Already visited
            return false;
        }
        vis(lv);
        return true;
    });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
