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
/*!
 * Interface for accessing the volume graph.
 * \tparam V Lightweight volume reference
 * \tparam VI Lightweight volume instance reference
 */
template<class V, class VI>
class VolumeAccessorInterface
{
  public:
    //! A lightweight identifier for a volume: OpaqueId or pointer
    using VolumeRef = V;
    //! A lightweight identifier for a volume instance
    using VolumeInstanceRef = VI;
    //! Result vector
    using VecVolInstRef = std::vector<VolumeInstanceRef>;

  public:
    //! Outgoing volume node from an instance
    virtual VolumeRef volume(VolumeInstanceRef parent) = 0;
    //! Outgoing instance nodes from a volume
    virtual VecVolInstRef children(VolumeRef parent) = 0;

  protected:
    ~VolumeAccessorInterface() = default;
};

//---------------------------------------------------------------------------//
/*!
 * Recursively walk through all unique volume instances.
 * \tparam VA Helper class with the same signature as VolumeAccessor above.
 *
 * This class can be used for both Geant4 and VecGeom to give the same visiting
 * behavior across the two. The volume accessor should have the same signature
 * as \c VolumeAccessor above.
 *
 * The visitor function must have the signature
 * <code>bool(*)(VolumeInstanceRef, int)</code>
 * where the return value indicates whether the volume's children should be
 * visited, and the integer is the depth of the volume being visited.
 *
 * By default this will visit all unique instances, i.e. every path in the
 * graph (the entire "touchable" hierarchy): this may be
 * very expensive! If it's desired to only visit single physical volumes, mark
 * them as visited using a set (see unit test for example) and return \c false
 * from the visitor to terminate the search path.
 */
template<class VA>
class VolumeInstanceVisitor
{
  public:
    using VolumeInstanceRef = typename VA::VolumeInstanceRef;

    //! Construct from accessor for obtaining volumes
    explicit VolumeInstanceVisitor(VA va) : accessor_(std::forward<VA>(va)) {}

    // Visit and return whether to continue visiting
    template<class F>
    inline void operator()(F&& visit, VolumeInstanceRef world);

  private:
    VA accessor_;

    struct QueuedVolume
    {
        VolumeInstanceRef vi{};
        int depth{0};
    };
};

//---------------------------------------------------------------------------//
/*!
 * Recursively visit volume instances.
 */
template<class VA>
class VolumeVisitor
{
  public:
    using VolumeRef = typename VA::VolumeRef;
    using VolumeInstanceRef = typename VA::VolumeInstanceRef;

    //! Construct from accessor for obtaining children
    explicit VolumeVisitor(VA va) : accessor_(std::forward<VA>(va)) {}

    // Apply this visitor
    template<class F>
    inline void operator()(F&& visit, VolumeInstanceRef top);

  private:
    VA accessor_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply the given visitor to the volume instance.
 *
 * Future work: we could keep full paths instead of just the depth if we
 * wanted.
 */
template<class VA>
template<class F>
void VolumeInstanceVisitor<VA>::operator()(F&& visit, VolumeInstanceRef world)
{
    std::vector<QueuedVolume> queue;
    std::vector<VolumeInstanceRef> temp_children;
    auto visit_impl = [&](VolumeInstanceRef vi, int depth) {
        if (visit(vi, depth))
        {
            auto vol = accessor_.volume(vi);
            auto&& children = accessor_.children(vol);
            // Append children in *reverse* order since we pop back
            for (auto iter = children.rbegin(); iter != children.rend(); ++iter)
            {
                queue.push_back({*iter, depth + 1});
            }
        }
    };

    // Visit the top-level physical volume
    visit_impl(world, 0);

    while (!queue.empty())
    {
        QueuedVolume qv = queue.back();
        queue.pop_back();

        // Visit popped daughter
        visit_impl(qv.vi, qv.depth);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Visit all logical volumes, once, depth-first.
 */
template<class VA>
template<class F>
void VolumeVisitor<VA>::operator()(F&& visit_vol, VolumeInstanceRef world)
{
    // Keep track of visited volumes
    std::unordered_set<VolumeRef> visited;
    // Convert volume instances to volumes, and
    auto visitor = [&](VolumeInstanceRef vi, int) -> bool {
        auto vol = accessor_.volume(vi);
        if (!visited.insert(vol).second)
        {
            // Already visited
            return false;
        }
        // Call user-supplied function and continue visiting children
        visit_vol(vol);
        return true;
    };

    VolumeInstanceVisitor visit_vol_inst{accessor_};
    visit_vol_inst(visitor, world);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
