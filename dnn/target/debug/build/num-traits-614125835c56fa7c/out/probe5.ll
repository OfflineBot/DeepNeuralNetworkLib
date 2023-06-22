; ModuleID = 'probe5.7e80b742-cgu.0'
source_filename = "probe5.7e80b742-cgu.0"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@alloc_a47d3ebcdca7ef1b1319c8eceabec8ae = private unnamed_addr constant <{ [75 x i8] }> <{ [75 x i8] c"/rustc/1716932743a7b3705cbf0c34db0c4e070ed1930d\\library\\core\\src\\num\\mod.rs" }>, align 1
@alloc_fcc48957c7b26b7f099f81b91ceae798 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_a47d3ebcdca7ef1b1319c8eceabec8ae, [16 x i8] c"K\00\00\00\00\00\00\00/\04\00\00\05\00\00\00" }>, align 8
@str.0 = internal constant [25 x i8] c"attempt to divide by zero"

; probe5::probe
; Function Attrs: uwtable
define void @_ZN6probe55probe17h41f5346ae74e6141E() unnamed_addr #0 {
start:
  %0 = call i1 @llvm.expect.i1(i1 false, i1 false)
  br i1 %0, label %panic.i, label %"_ZN4core3num21_$LT$impl$u20$u32$GT$10div_euclid17h7829a11d5d745821E.exit"

panic.i:                                          ; preds = %start
; call core::panicking::panic
  call void @_ZN4core9panicking5panic17h143ba3a32d7a8901E(ptr align 1 @str.0, i64 25, ptr align 8 @alloc_fcc48957c7b26b7f099f81b91ceae798) #3
  unreachable

"_ZN4core3num21_$LT$impl$u20$u32$GT$10div_euclid17h7829a11d5d745821E.exit": ; preds = %start
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind readnone willreturn
declare i1 @llvm.expect.i1(i1, i1) #1

; core::panicking::panic
; Function Attrs: cold noinline noreturn uwtable
declare void @_ZN4core9panicking5panic17h143ba3a32d7a8901E(ptr align 1, i64, ptr align 8) unnamed_addr #2

attributes #0 = { uwtable "target-cpu"="x86-64" }
attributes #1 = { nocallback nofree nosync nounwind readnone willreturn }
attributes #2 = { cold noinline noreturn uwtable "target-cpu"="x86-64" }
attributes #3 = { noreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"PIC Level", i32 2}
