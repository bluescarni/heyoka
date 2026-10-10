Multidimensional array views and associated utilities
=====================================================

.. cpp:namespace-push:: heyoka

*#include <heyoka/mdspan.hpp>*

In order to interact with and represent multidimensional array views,
heyoka includes a `reference implementation <https://github.com/kokkos/mdspan>`__
of the `mdspan <https://en.cppreference.com/w/cpp/container/mdspan>`__
class from C++23/C++26, plus some of the associated utilities. The implementation in heyoka
follows the standard API and, as an extension, it also supports indexing via the
function call operator. Note that the classes and functions listed below are distinct from
those in the ``std`` namespace, even if the compiler provides them.

Please refer to the `documentation page on cppreference <https://en.cppreference.com/w/cpp/header/mdspan>`__
for the full API documentation.

.. cpp:type:: template <typename T, typename Extents, typename LayoutPolicy = layout_right, typename AccessorPolicy = default_accessor<T> > mdspan

.. cpp:type:: template <typename IndexType, std::size_t... Extents> extents

.. cpp:type:: template <typename IndexType, std::size_t Rank> dextents

.. cpp:class:: layout_left
               layout_right
               layout_stride

.. cpp:type:: template <typename ElementType> default_accessor

.. cpp:var:: inline constexpr auto full_extent

.. cpp:function:: template <typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy, typename... SliceSpecifiers> constexpr auto submdspan(const mdspan<T, Extents, LayoutPolicy, AccessorPolicy> &x, SliceSpecifiers... slices)
