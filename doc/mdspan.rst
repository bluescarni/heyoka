Multidimensional array views
============================

.. cpp:namespace-push:: heyoka

*#include <heyoka/mdspan.hpp>*

In order to interact with and represent multidimensional array views,
heyoka includes a `reference implementation <https://github.com/kokkos/mdspan>`__
of the `mdspan <https://en.cppreference.com/w/cpp/container/mdspan>`__
class from C++23. The implementation in heyoka is fully standard-compliant,
except for the fact that it supports as an extension indexing via the
function call operator.

.. cpp:type:: template <typename T, typename Extents, typename LayoutPolicy = std::experimental::layout_right, typename AccessorPolicy = std::experimental::default_accessor<T> > mdspan = std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>

   Multidimensional array view.

   See the `mdspan <https://en.cppreference.com/w/cpp/container/mdspan>`__ documentation
   and `tutorial <https://github.com/kokkos/mdspan/wiki/A-Gentle-Introduction-to-mdspan>`__.

.. cpp:type:: template <typename IndexType, std::size_t... Extents> extents = std::experimental::extents<IndexType, Extents...>
              template <typename IndexType, std::size_t Rank> dextents = std::experimental::dextents<IndexType, Rank>

   Classes representing static and dynamic extents.

   See the `documentation <https://en.cppreference.com/w/cpp/container/mdspan/extents>`__.
